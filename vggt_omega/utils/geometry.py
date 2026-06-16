# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def unproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject depth maps into camera-frame 3D points -> (..., H, W, 3).

    Differentiable torch twin of ``inference.unproject_depth_map_to_point_map``:
    integer pixel-grid coordinates, OpenCV pinhole model.

    Args:
        depth: (..., H, W) depth values.
        intrinsics: (..., 3, 3) pinhole matrices, batch dims matching ``depth``.
    """
    height, width = depth.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )

    batch_shape = depth.shape[:-2]
    view = batch_shape + (1, 1)
    fx = intrinsics[..., 0, 0].reshape(view)
    fy = intrinsics[..., 1, 1].reshape(view)
    cx = intrinsics[..., 0, 2].reshape(view)
    cy = intrinsics[..., 1, 2].reshape(view)

    return torch.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], dim=-1)


def transform_points(extrinsics: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply ``(..., 3, 4)`` [R|t] transforms to ``(..., H, W, 3)`` points."""
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return torch.einsum("...ij,...hwj->...hwi", rotation, points) + translation[..., None, None, :]


def invert_transform_points(extrinsics: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply the inverse of ``(..., 3, 4)`` [R|t] transforms to ``(..., H, W, 3)`` points."""
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return torch.einsum("...ji,...hwj->...hwi", rotation, points - translation[..., None, None, :])


def compose_with_inverse(extrinsics: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Compose ``extrinsics ∘ inv(reference)`` for ``(..., 3, 4)`` [R|t] matrices.

    With camera-from-world inputs and ``reference`` the first camera, the result
    is camera-from-reference: the same poses re-expressed in the reference
    camera's coordinate frame.
    """
    R = extrinsics[..., :3, :3]
    T = extrinsics[..., :3, 3]
    R_ref = reference[..., :3, :3]
    T_ref = reference[..., :3, 3]

    R_rel = torch.einsum("...ij,...kj->...ik", R, R_ref)
    T_rel = T - torch.einsum("...ij,...j->...i", R_rel, T_ref)
    return torch.cat([R_rel, T_rel[..., None]], dim=-1)


def closed_form_inverse_se3(se3, R=None, T=None):
    """Invert a batch of 3x4 or 4x4 SE(3) matrices."""
    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must have shape (N, 4, 4) or (N, 3, 4), got {se3.shape}")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_t = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_t, T)
        inverted = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_t = R.transpose(1, 2)
        top_right = -torch.bmm(R_t, T)
        inverted = torch.eye(4, device=R.device, dtype=R.dtype)[None].repeat(len(R), 1, 1)

    inverted[:, :3, :3] = R_t
    inverted[:, :3, 3:] = top_right
    return inverted


def project_world_points_to_cam(world_points, extrinsics, intrinsics):
    """Project P world points into N cameras.

    Args:
        world_points: (P, 3) tensor.
        extrinsics: (N, 3, 4) world-to-camera OpenCV [R|t].
        intrinsics: (N, 3, 3) pinhole K in pixels.

    Returns:
        image_points: (N, P, 2) pixel coords.
        cam_points: (N, 3, P) camera-frame points (row 2 = depth).
    """
    R = extrinsics[:, :3, :3]                       # (N,3,3)
    t = extrinsics[:, :3, 3:]                       # (N,3,1)
    cam_points = R @ world_points.T.unsqueeze(0) + t    # (N,3,P)
    uvw = intrinsics @ cam_points                   # (N,3,P)
    z = uvw[:, 2:].clamp(min=1e-8)
    image_points = (uvw[:, :2] / z).transpose(1, 2)     # (N,P,2)
    return image_points, cam_points


def cam_from_img(tracks, intrinsics):
    """Pixel -> normalized camera coords. tracks (N,P,2), intrinsics (N,3,3) -> (N,P,2)."""
    fx = intrinsics[:, 0, 0].unsqueeze(-1); fy = intrinsics[:, 1, 1].unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1); cy = intrinsics[:, 1, 2].unsqueeze(-1)
    return torch.stack([(tracks[..., 0] - cx) / fx, (tracks[..., 1] - cy) / fy], dim=-1)


def sampson_epipolar_distance(pts1, pts2, Fm, eps: float = 1e-8):
    """Sampson distance for point pairs under fundamental/essential matrices.

    pts1, pts2: (N, P, 2); Fm: (N, 3, 3). Returns (N, P).
    Matches kornia.geometry.epipolar.sampson_epipolar_distance:
    (x2^T F x1)^2 / (||(F x1)_{0:2}||^2 + ||(F^T x2)_{0:2}||^2).
    """
    ones = torch.ones_like(pts1[..., :1])
    x1 = torch.cat([pts1, ones], dim=-1)            # (N,P,3)
    x2 = torch.cat([pts2, ones], dim=-1)
    Fx1 = x1 @ Fm.transpose(1, 2)                   # (N,P,3) rows = F @ x1
    Ftx2 = x2 @ Fm                                  # (N,P,3) rows = F^T @ x2
    num = (x2 * Fx1).sum(dim=-1) ** 2
    den = Fx1[..., 0] ** 2 + Fx1[..., 1] ** 2 + Ftx2[..., 0] ** 2 + Ftx2[..., 1] ** 2
    return num / (den + eps)
