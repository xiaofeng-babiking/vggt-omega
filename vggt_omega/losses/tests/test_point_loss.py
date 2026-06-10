"""Tests for the point loss against analytically-known values.

Predicted cameras decode through ``encoding_to_camera``, which puts the
principal point at the image center, so GT scenes here use centered
intrinsics to make exact round trips possible. With identity cameras the
unprojection lives in the first-camera frame directly; translating the
predicted camera by ``δ`` shifts every predicted point by ``−δ`` (a constant
3D residual of norm ``|δ|`` with zero spatial gradient).
"""

from __future__ import annotations

import pytest
import torch

from vggt_omega.losses import point_loss, predict_point_map
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding

H = W = 4


def _centered_intrinsics(f: float = 2.0) -> torch.Tensor:
    return torch.tensor([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]])


def _identity_extrinsic() -> torch.Tensor:
    return torch.cat([torch.eye(3), torch.zeros(3, 1)], dim=-1)


def _gt_points(depth: torch.Tensor, intrinsics: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Unproject through [I|t] cameras into the reference frame, by hand."""
    y, x = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    cam = torch.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], dim=-1)
    return cam - translation


def test_perfect_prediction_is_zero():
    intrinsics = _centered_intrinsics()
    extrinsics = _identity_extrinsic()[None, None]
    depth = torch.full((1, 1, H, W), 2.0)
    pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics[None, None], (H, W))
    gt_points = _gt_points(depth[0, 0], intrinsics, torch.zeros(3))[None, None]
    valid = torch.ones_like(depth, dtype=torch.bool)

    loss = point_loss(depth, torch.ones_like(depth), pose_enc, gt_points, depth, valid, alpha=0.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_translated_camera_round_trips():
    # GT camera [I | (0, 0, 1)]: a perfect prediction must still be zero,
    # validating the cam->reference transform inside the unprojection.
    intrinsics = _centered_intrinsics()
    translation = torch.tensor([0.0, 0.0, 1.0])
    extrinsics = torch.cat([torch.eye(3), translation[:, None]], dim=-1)[None, None]
    depth = torch.full((1, 1, H, W), 2.0)
    pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics[None, None], (H, W))
    gt_points = _gt_points(depth[0, 0], intrinsics, translation)[None, None]
    valid = torch.ones_like(depth, dtype=torch.bool)

    loss = point_loss(depth, torch.ones_like(depth), pose_enc, gt_points, depth, valid, alpha=0.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_camera_translation_offset_value():
    # Shifting the predicted camera by delta in x moves every predicted point
    # by -delta: constant residual norm delta, weight 1 + 1/2 = 1.5 at D=2.
    delta = 0.4
    intrinsics = _centered_intrinsics()
    extrinsics = _identity_extrinsic()[None, None]
    depth = torch.full((1, 1, H, W), 2.0)
    pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics[None, None], (H, W))
    pose_enc[..., 0] += delta
    gt_points = _gt_points(depth[0, 0], intrinsics, torch.zeros(3))[None, None]
    valid = torch.ones_like(depth, dtype=torch.bool)

    loss = point_loss(depth, torch.ones_like(depth), pose_enc, gt_points, depth, valid, alpha=0.0)
    assert loss.item() == pytest.approx(1.5 * delta, rel=1e-5)


def test_depth_error_scales_point_residual():
    # With centered intrinsics the pixel at the principal point unprojects to
    # (0, 0, depth): a depth error of delta there is a 3D error of norm delta.
    intrinsics = _centered_intrinsics()
    extrinsics = _identity_extrinsic()[None, None]
    pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics[None, None], (H, W))
    depth = torch.full((1, 1, H, W), 1.0)
    pred_depth = depth.clone()
    pred_depth[0, 0, H // 2, W // 2] += 0.3
    gt_points = _gt_points(depth[0, 0], intrinsics, torch.zeros(3))[None, None]
    valid = torch.zeros_like(depth, dtype=torch.bool)
    valid[0, 0, H // 2, W // 2] = True  # isolate the centre pixel

    loss = point_loss(pred_depth, torch.ones_like(depth), pose_enc, gt_points, depth, valid, alpha=0.0)
    assert loss.item() == pytest.approx(2.0 * 0.3, rel=1e-5)


def test_predict_point_map_gradients_flow():
    intrinsics = _centered_intrinsics()
    extrinsics = _identity_extrinsic()[None, None]
    depth = torch.full((1, 1, H, W, 1), 2.0, requires_grad=True)
    pose_enc = extri_intri_to_pose_encoding(extrinsics, intrinsics[None, None], (H, W))
    pose_enc.requires_grad_(True)

    points = predict_point_map(depth, pose_enc, (H, W))
    points.sum().backward()

    assert depth.grad is not None and torch.isfinite(depth.grad).all()
    assert pose_enc.grad is not None and torch.isfinite(pose_enc.grad).all()
