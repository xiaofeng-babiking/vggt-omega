# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Ground-truth unit-space normalization (VGGT-Omega paper, Sec. A.1).

Per scene, all ground-truth quantities are re-expressed in the first camera's
coordinate frame, and depth maps and translation vectors are divided by the
average distance of all valid 3D points to the origin. The normalization is
applied only to the ground truth, never to the predictions.
"""

from dataclasses import dataclass

import torch

from vggt_omega.utils.geometry import compose_with_inverse, invert_transform_points, unproject_depth


@dataclass
class NormalizedScene:
    """Ground truth normalized to the first camera's unit coordinate space.

    Attributes:
        extrinsics: (B, S, 3, 4) camera-from-first-camera [R|t]; frame 0 is identity.
        depths: (B, S, H, W) scaled GT depths (invalid pixels are zeroed).
        points: (B, S, H, W, 3) per-pixel GT 3D points in the first camera's frame.
        valid_mask: (B, S, H, W) bool, pixels with usable GT depth.
        scale: (B,) the per-scene divisor (average point distance to the origin).
    """

    extrinsics: torch.Tensor
    depths: torch.Tensor
    points: torch.Tensor
    valid_mask: torch.Tensor
    scale: torch.Tensor


def normalize_scene_to_first_camera(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    depths: torch.Tensor,
    point_masks: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> NormalizedScene:
    """Normalize GT cameras/depths to the first camera's unit coordinate space.

    Args:
        extrinsics: (B, S, 3, 4) camera-from-world [R|t], OpenCV convention.
        intrinsics: (B, S, 3, 3) pinhole matrices.
        depths: (B, S, H, W); 0 = invalid, < 0 = sky (dataset convention).
        point_masks: optional (B, S, H, W) bool of additionally-valid pixels.
        eps: lower clamp for the scale divisor.
    """
    extrinsics = extrinsics.float()
    intrinsics = intrinsics.float()
    depths = depths.float()

    valid = torch.isfinite(depths) & (depths > 0)
    if point_masks is not None:
        valid = valid & point_masks.bool()
    # Zero invalid depths so non-finite GT cannot poison the scale reduction
    # (inf * 0 = NaN) or leak NaN into downstream masked means and gradients.
    depths = torch.where(valid, depths, torch.zeros_like(depths))

    extrinsics_rel = compose_with_inverse(extrinsics, extrinsics[:, :1])

    cam_points = unproject_depth(depths, intrinsics)
    points = invert_transform_points(extrinsics_rel, cam_points)

    distances = points.norm(dim=-1)
    valid_f = valid.float()
    num_valid = valid_f.sum(dim=(1, 2, 3))
    scale = (distances * valid_f).sum(dim=(1, 2, 3)) / num_valid.clamp(min=1)
    # Scenes with no valid GT keep scale 1 so downstream terms stay finite.
    scale = torch.where(num_valid > 0, scale, torch.ones_like(scale)).clamp(min=eps)

    inv_scale = 1.0 / scale
    extrinsics_scaled = extrinsics_rel.clone()
    extrinsics_scaled[..., 3] = extrinsics_rel[..., 3] * inv_scale.view(-1, 1, 1)
    return NormalizedScene(
        extrinsics=extrinsics_scaled,
        depths=depths * inv_scale.view(-1, 1, 1, 1),
        points=points * inv_scale.view(-1, 1, 1, 1, 1),
        valid_mask=valid,
        scale=scale,
    )
