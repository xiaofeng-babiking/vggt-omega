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


def test_depth_loss_gradient_term_sees_sign_flips():
    # e = [+1, -1] on adjacent pixels: |grad e| = 2 but grad|e| = 0 — the loss
    # must differentiate the signed residual (paper: c * grad(e), e = pred - gt).
    gt = torch.full((1, 1, 1, 2), 5.0)
    pred = gt + torch.tensor([1.0, -1.0]).view(1, 1, 1, 2)
    conf = torch.ones_like(gt)
    valid = torch.ones_like(gt, dtype=torch.bool)
    loss = depth_loss(pred, conf, gt, valid, alpha=0.0)
    # data term: mean((1 + 1/5) * 1) = 1.2; gradient term: |(-1) - (+1)| = 2
    assert torch.allclose(loss, torch.tensor(3.2), atol=1e-6)


# --- Exact-value tests (pin conventions; would catch the camera-mean and
# point-L1 regressions that ordering-only checks miss) ---

def test_camera_loss_is_per_frame_l1_norm_not_per_component_mean():
    # pred-gt = 0.1 on every one of the 9 components, B=1 S=2.
    # Paper L_cam = mean_BS( sum_9 |.| ) = 0.1*9 = 0.9.  The per-component mean
    # (the old bug) would give 0.1 — exactly 1/9.
    gt = torch.zeros(1, 2, 9)
    pred = gt + 0.1
    assert torch.allclose(camera_loss(pred, gt), torch.tensor(0.9), atol=1e-6)


def test_point_loss_residual_is_l2_norm_over_xyz():
    # One frame, 1x1 image, pred point off GT by (3,4,0): L2 = 5, L1 would be 7.
    # conf=1, alpha=0, gt_depth=1 -> data weight (1+1/1)=2 -> data term = 2*5 = 10.
    # Build a pred depth+cam that unprojects to gt+(3,4,0). Simpler: drive the
    # shared _aleatoric_terms via depth_loss with a 3-channel residual is not
    # exposed, so assert the magnitude convention through point_loss end-to-end
    # is L2 by comparing an axis-spread error to an axis-aligned one of equal L1.
    import torch as _t
    from vggt_omega.training.losses import _aleatoric_terms
    valid = _t.ones(1, 1, 1, 1, dtype=_t.bool)
    gt_depth = _t.ones(1, 1, 1, 1)
    conf = _t.ones(1, 1, 1, 1)
    spread = _t.tensor([3.0, 4.0, 0.0]).reshape(1, 1, 1, 1, 3)   # L2=5, L1=7
    axis = _t.tensor([5.0, 0.0, 0.0]).reshape(1, 1, 1, 1, 3)     # L2=5, L1=5
    l_spread = _aleatoric_terms(spread, conf, gt_depth, valid, alpha=0.0)
    l_axis = _aleatoric_terms(axis, conf, gt_depth, valid, alpha=0.0)
    assert torch.allclose(l_spread, l_axis, atol=1e-6)           # equal L2 -> equal loss
    assert torch.allclose(l_spread, torch.tensor(10.0), atol=1e-6)  # 2*(1+1/1... )=2*5


def test_aleatoric_data_term_weight_and_alpha_exact():
    from vggt_omega.training.losses import _aleatoric_terms
    # depth residual e=0.5 everywhere, gt_depth D=2 -> weight (1+1/2)=1.5,
    # conf c=2 -> data = c*w*|e| = 2*1.5*0.5 = 1.5; grad term 0 (uniform e);
    # reg = -alpha*log(c). With alpha=0.2: reg = -0.2*log(2).
    e = torch.full((1, 1, 2, 2), 0.5)
    D = torch.full((1, 1, 2, 2), 2.0)
    c = torch.full((1, 1, 2, 2), 2.0)
    valid = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    import math
    expected = 1.5 + 0.0 - 0.2 * math.log(2.0)
    assert torch.allclose(_aleatoric_terms(e, c, D, valid, alpha=0.2),
                          torch.tensor(expected), atol=1e-6)
