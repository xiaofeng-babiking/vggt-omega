# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Point loss (VGGT-Omega paper, Sec. 3.2).

Identical to the depth loss up to the residual: ``e = π⁻¹(D̂, ĝ) − P``, where
the *predicted* depth is unprojected through the *predicted* camera into the
first camera's frame and compared with the GT point map. This supervises point
maps without a dedicated dense head, since both depth and camera gradients
flow through the unprojection.
"""

import torch

from vggt_omega.utils.geometry import invert_transform_points, unproject_depth
from vggt_omega.utils.pose_enc import encoding_to_camera

from .regression import confidence_weighted_loss


def predict_point_map(
    pred_depth: torch.Tensor,
    pred_pose_enc: torch.Tensor,
    image_size_hw: tuple[int, int],
) -> torch.Tensor:
    """Unproject predicted depth via the predicted cameras -> (B, S, H, W, 3).

    Points land in the first camera's frame because VGGT-Omega predicts
    cameras relative to frame 0. Differentiable in both inputs.
    """
    if pred_depth.dim() == 5:
        pred_depth = pred_depth.squeeze(-1)
    extrinsics, intrinsics = encoding_to_camera(pred_pose_enc.float(), image_size_hw)
    cam_points = unproject_depth(pred_depth.float(), intrinsics)
    return invert_transform_points(extrinsics, cam_points)


def point_loss(
    pred_depth: torch.Tensor,
    pred_conf: torch.Tensor,
    pred_pose_enc: torch.Tensor,
    gt_points: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Confidence-weighted point-map loss.

    Args:
        pred_depth: (B, S, H, W, 1) or (B, S, H, W) predicted depth.
        pred_conf: (B, S, H, W) predicted confidence, ≥ 1 (shared with depth).
        pred_pose_enc: (B, S, 9) predicted pose encoding.
        gt_points: (B, S, H, W, 3) GT points in the first camera's frame,
            normalized to unit space.
        gt_depth: (B, S, H, W) normalized GT depth for the (1 + 1/D) factor.
        valid: (B, S, H, W) bool mask of pixels with usable GT.
        alpha: weight of the −log c uncertainty term (see ``regression``).
    """
    height, width = gt_depth.shape[-2:]
    pred_points = predict_point_map(pred_depth, pred_pose_enc, (height, width))
    residual = pred_points - gt_points.float()
    return confidence_weighted_loss(residual, pred_conf, gt_depth, valid, alpha=alpha)
