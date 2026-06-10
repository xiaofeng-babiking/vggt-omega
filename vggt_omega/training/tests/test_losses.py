import pytest
import torch

from vggt_omega.training.losses import (
    camera_loss,
    depth_loss,
    normalize_gt_into_first_camera,
    point_loss,
)
from vggt_omega.training.tests.conftest import (
    SCENE_H,
    SCENE_W,
    _intrinsics_for_scene,
    _random_consistent_scene,
)
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


@pytest.fixture
def scene():
    ext, dep, wp, mask = _random_consistent_scene(B=1, S=3)
    n_ext, n_dep, n_wp, scale = normalize_gt_into_first_camera(ext, dep, wp, mask)
    K = _intrinsics_for_scene(B=1, S=3)
    gt_enc = extri_intri_to_pose_encoding(n_ext, K, (SCENE_H, SCENE_W))
    return dict(gt_enc=gt_enc, gt_depth=n_dep, gt_points=n_wp, valid=mask)


def test_camera_loss_zero_at_gt(scene):
    gt_enc = scene["gt_enc"]
    assert camera_loss(gt_enc, gt_enc) == 0


def test_camera_loss_positive_and_decreasing(scene):
    gt_enc = scene["gt_enc"]
    noisy = gt_enc + 0.1
    less_noisy = gt_enc + 0.01
    assert camera_loss(noisy, gt_enc) > camera_loss(less_noisy, gt_enc) > 0


def test_depth_loss_floor_at_gt(scene):
    gt_depth, valid = scene["gt_depth"], scene["valid"]
    conf = torch.ones_like(gt_depth)
    l_perfect = depth_loss(gt_depth, conf, gt_depth, valid, alpha=0.2)
    assert torch.allclose(l_perfect, torch.tensor(0.0), atol=1e-6)
    pred = gt_depth * 1.2
    assert depth_loss(pred, conf, gt_depth, valid, alpha=0.2) > 0.01


def test_depth_loss_invalid_pixels_no_gradient(scene):
    gt_depth, valid = scene["gt_depth"], scene["valid"]
    pred = gt_depth.clone().requires_grad_(True)
    mask = valid.clone()
    mask[..., :8] = False
    depth_loss(pred * 1.1, torch.ones_like(gt_depth) * 2, gt_depth, mask).backward()
    assert (pred.grad[..., :8] == 0).all()
    assert torch.isfinite(pred.grad).all()


def test_depth_loss_confidence_tradeoff(scene):
    gt_depth, valid = scene["gt_depth"], scene["valid"]
    pred = gt_depth * 1.5
    hi = depth_loss(pred, torch.full_like(gt_depth, 3.0), gt_depth, valid)
    lo = depth_loss(pred, torch.full_like(gt_depth, 1.01), gt_depth, valid)
    assert hi > lo


def test_point_loss_zero_at_gt(scene):
    gt_enc, n_dep, n_wp, valid = (
        scene["gt_enc"],
        scene["gt_depth"],
        scene["gt_points"],
        scene["valid"],
    )
    l = point_loss(
        n_dep, torch.ones_like(n_dep), gt_enc, n_wp, n_dep, valid, (SCENE_H, SCENE_W)
    )
    assert torch.allclose(l, torch.tensor(0.0), atol=1e-5)


def test_point_loss_penalizes_wrong_pose(scene):
    gt_enc, n_dep, n_wp, valid = (
        scene["gt_enc"],
        scene["gt_depth"],
        scene["gt_points"],
        scene["valid"],
    )
    bad = gt_enc.clone()
    bad[..., 0] += 0.2
    l = point_loss(
        n_dep, torch.ones_like(n_dep), bad, n_wp, n_dep, valid, (SCENE_H, SCENE_W)
    )
    assert l > 0.01
