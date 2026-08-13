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


# Relative second-singular-value floor below which a centred point set is treated
# as rank-deficient (coincident or colinear cameras) and no rotation is fitted.
_SIM3_RANK_EPS = 1e-6


def camera_centers(extrinsics: torch.Tensor) -> torch.Tensor:
    """World-space camera centres from ``(..., 3, 4)`` camera-from-world [R|t].

    ``c = -Rᵀ t`` — the world point that maps to the camera origin.
    """
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return -torch.einsum("...ji,...j->...i", rotation, translation)


def _baseline_ratio(src_centred: torch.Tensor, dst_centred: torch.Tensor) -> torch.Tensor:
    """Median ratio of centred point norms: a rotation-free scale estimate.

    Well defined for degenerate sets where the Umeyama SVD is not. Returns 1.0
    when the source points carry no spread at all.
    """
    src_norm = src_centred.norm(dim=-1)
    dst_norm = dst_centred.norm(dim=-1)
    keep = src_norm > 1e-12
    if not bool(keep.any()):
        return torch.ones((), dtype=src_centred.dtype, device=src_centred.device)
    return (dst_norm[keep] / src_norm[keep]).median()


def umeyama_sim3(src: torch.Tensor, dst: torch.Tensor):
    """Least-squares similarity transform taking ``src`` onto ``dst``.

    Umeyama (1991) closed form for ``dst ≈ scale · rotation @ src + translation``
    over corresponding ``(M, 3)`` point sets. Computed in float64 on CPU — the
    covariance SVD is ill-conditioned in fp32 for the near-identical trajectories
    this is used on, and doing the (tiny, M ≤ ~100 points) linear algebra on CPU
    keeps it off the accelerator's spottier fp64 kernel coverage. The only caller
    is already inside @torch.no_grad(). Results are returned in the input dtype
    and on the input device.

    Degenerate sources (fewer than three points, or a centred set of rank < 2:
    coincident or colinear camera centres) do not determine a rotation, and a
    plain SVD would return an arbitrary one that tumbles the whole trajectory.
    Those fall back to ``rotation = I`` with a median-baseline scale. Empty input
    (M=0) returns identity rotation, unit scale, and zero translation.

    Returns ``(scale (), rotation (3, 3), translation (3,))``.
    """
    dtype = src.dtype
    device = src.device

    # Empty input: return identity Sim(3).
    if src.shape[0] == 0:
        return (
            torch.ones((), dtype=dtype, device=device),
            torch.eye(3, dtype=dtype, device=device),
            torch.zeros(3, dtype=dtype, device=device),
        )

    # Do all float64 linear algebra on CPU.
    src_cpu = src.cpu().double()
    dst_cpu = dst.cpu().double()
    mean_src = src_cpu.mean(dim=0)
    mean_dst = dst_cpu.mean(dim=0)
    src_c = src_cpu - mean_src
    dst_c = dst_cpu - mean_dst
    variance = (src_c**2).sum(dim=-1).mean()

    # Non-finite input is a THIRD kind of degeneracy, and it has to be caught
    # before either SVD below: torch.linalg.svd (and svdvals) RAISE
    # _LinAlgError on non-finite input rather than returning something the
    # checks further down could reject. Every other degeneracy here -- too few
    # points, zero variance, collinear centres, a reflected rotation, a
    # non-positive scale -- is handled by falling back to identity, and only
    # this one killed the process (a drifting student emitting a non-finite
    # pose, inside a @torch.no_grad CPU helper that cannot affect a gradient).
    #
    # Falls back like the others. The caller is expected to notice: with
    # non-finite input the residual computed against it is non-finite too, and
    # sim3_refuse_gate refuses on `value.isfinite()`, so the held-out views are
    # dropped for that item rather than scored against a meaningless transform.
    finite_input = bool(torch.isfinite(src_cpu).all() and torch.isfinite(dst_cpu).all())
    degenerate = not finite_input or src_cpu.shape[0] < 3 or not bool(variance > 1e-12)
    if not degenerate:
        singular = torch.linalg.svdvals(src_c)
        degenerate = bool(singular[1] <= _SIM3_RANK_EPS * singular[0])

    if degenerate:
        scale = _baseline_ratio(src_c, dst_c)
        rotation = torch.eye(3, dtype=torch.float64, device="cpu")
    else:
        cov = (dst_c.t() @ src_c) / src_cpu.shape[0]
        U, D, Vh = torch.linalg.svd(cov)
        # Reflection guard: an unconstrained SVD can return det = -1, a mirror.
        correction = torch.ones(3, dtype=torch.float64, device="cpu")
        if torch.det(U) * torch.det(Vh) < 0:
            correction[-1] = -1.0
        rotation = U @ torch.diag(correction) @ Vh
        scale = (D * correction).sum() / variance
        if not bool(torch.isfinite(scale)) or not bool(scale > 0):
            scale = _baseline_ratio(src_c, dst_c)
            rotation = torch.eye(3, dtype=torch.float64, device="cpu")

    translation = mean_dst - scale * (rotation @ mean_src)

    # Convert to input dtype and device.
    return (
        scale.to(dtype=dtype, device=device),
        rotation.to(dtype=dtype, device=device),
        translation.to(dtype=dtype, device=device),
    )


def apply_sim3_to_extrinsics(
    extrinsics: torch.Tensor,
    scale: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Re-express camera-from-world ``(..., 3, 4)`` poses under a Sim(3) on the world.

    ``scale, rotation, translation`` move world points as ``p' = s · R @ p + t``
    (the :func:`umeyama_sim3` convention). A camera with centre ``c`` and
    orientation ``R_cam`` moves to centre ``s · R @ c + t`` with orientation
    ``R_cam @ Rᵀ``.

    The exact similarity-consistent camera matrix carries a ``1/s`` factor, which
    is dropped here: it scales camera-space coordinates uniformly, so it changes
    rendered depth but not the projected image. Dropping it keeps the returned
    rotation block exactly orthonormal — which matters, because fp32 pose-head
    output already drifts off SO(3) and the eval code has to re-project it back.
    """
    rot_cam = extrinsics[..., :3, :3]
    centers = camera_centers(extrinsics)
    new_centers = scale * torch.einsum("ij,...j->...i", rotation, centers) + translation
    new_rot = torch.einsum("...ij,kj->...ik", rot_cam, rotation)
    new_trans = -torch.einsum("...ij,...j->...i", new_rot, new_centers)
    return torch.cat([new_rot, new_trans[..., None]], dim=-1)


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


# --- numpy depth/pose helpers -----------------------------------------------
# Canonical home of the numpy counterparts of the torch helpers above
# (inference_common re-exports them for its callers). Used by the inference
# entrypoint (PLY fusion, pose scoring) and the COLMAP dataset path; kept numpy
# because they run on CPU prediction arrays after the forward, not inside the
# autograd graph.

def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray
) -> np.ndarray:
    """Unproject per-frame depth into a common world frame -> (S, H, W, 3).

    `extrinsic` is world-to-camera (OpenCV) [R|t], so world = R^T @ (cam - t).
    """
    depth = depth_map[..., 0]
    num, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num, height, width))
    y = np.broadcast_to(y[None], (num, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def world_to_camera_to_camera_to_world(w2c: np.ndarray) -> np.ndarray:
    """Invert world-to-camera ``(S, 3, 4)`` -> camera-to-world ``(S, 4, 4)``.

    The translation column of the result is the camera centre in world coords --
    the trajectory positions :class:`CameraPoseMetric` consumes.
    """
    rotation = w2c[:, :3, :3]
    translation = w2c[:, :3, 3]
    rot_c2w = np.transpose(rotation, (0, 2, 1))
    trans_c2w = -np.einsum("sij,sj->si", rot_c2w, translation)
    c2w = np.tile(np.eye(4), (w2c.shape[0], 1, 1))
    c2w[:, :3, :3] = rot_c2w
    c2w[:, :3, 3] = trans_c2w
    return c2w
