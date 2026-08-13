"""Unit tests for the self-supervised distillation pieces (CPU, tiny tensors)."""
import math

import pytest
import torch

from vggt_omega.models.vggt_omega import VGGTOmega
from vggt_omega.training.selfsup import (
    DistillLossComputer,
    augment_two_views,
    freeze_geometry_heads,
    _feature_l2,
)
from vggt_omega.training.trainer import init_model_from_scratch


class _AugCfg:
    brightness = 0.4
    contrast = 0.4
    saturation = 0.4
    mask_ratio = 0.3
    mask_patch = 8


def _pred(B=1, S=2, N=6, C=8, H=8, W=8, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *sh: torch.randn(*sh, generator=g, requires_grad=grad)
    return {
        "features": [mk(B, S, N, C), mk(B, S, N, C)],
        "pose_enc": mk(B, S, 9),
        "depth": mk(B, S, H, W, 1),
        "depth_conf": torch.rand(B, S, H, W, generator=g),
    }


WEIGHTS = {"feature": 1.0, "camera": 1.0, "depth": 1.0}


def test_identical_student_teacher_is_zero_loss():
    t = _pred()
    # student == teacher: detach so the comparison is value-only.
    s = {"features": [f.clone() for f in t["features"]], "pose_enc": t["pose_enc"].clone(),
         "depth": t["depth"].clone(), "depth_conf": t["depth_conf"].clone()}
    out = DistillLossComputer(WEIGHTS)(s, t)
    for k in ("feature", "camera", "depth", "total"):
        assert out[k].abs().item() < 1e-6, k


def test_perturbed_student_is_positive():
    t = _pred()
    s = _pred(seed=1)
    out = DistillLossComputer(WEIGHTS)(s, t)
    for k in ("feature", "camera", "depth"):
        assert out[k].item() > 0, k


def test_total_is_weighted_sum():
    t = _pred()
    s = _pred(seed=1)
    w = {"feature": 2.0, "camera": 0.5, "depth": 0.0}
    out = DistillLossComputer(w)(s, t)
    expected = 2.0 * out["feature"] + 0.5 * out["camera"] + 0.0 * out["depth"]
    assert torch.allclose(out["total"], expected)


def test_gradient_flows_to_student_only():
    t = _pred()  # teacher: no grad (detached targets)
    s = _pred(seed=1, grad=True)
    out = DistillLossComputer(WEIGHTS)(s, t)
    out["total"].backward()
    assert all(f.grad is not None for f in s["features"])
    assert s["pose_enc"].grad is not None and s["depth"].grad is not None
    assert all(f.grad is None for f in t["features"])  # teacher never accumulates


def test_conf_weighting_changes_the_depth_term():
    t = _pred()
    s = _pred(seed=1)
    weighted = DistillLossComputer(WEIGHTS, conf_weighted_depth=True)(s, t)["depth"]
    plain = DistillLossComputer(WEIGHTS, conf_weighted_depth=False)(s, t)["depth"]
    assert not torch.allclose(weighted, plain)


def test_feature_layer_count_mismatch_raises():
    with pytest.raises(ValueError, match="architecture"):
        _feature_l2([torch.randn(1, 2, 6, 8)], [torch.randn(1, 2, 6, 8)] * 2)


def test_augment_two_views_preserves_shape_and_range_but_differs():
    imgs = torch.rand(1, 3, 3, 16, 16)  # (B,S,3,H,W) in [0,1]
    v_s, v_t = augment_two_views(imgs, _AugCfg())
    assert v_s.shape == imgs.shape == v_t.shape
    assert v_s.min() >= 0 and v_s.max() <= 1
    # Independent draws -> the two views must differ, and both differ from the input.
    assert not torch.allclose(v_s, v_t)
    assert not torch.allclose(v_s, imgs)


def test_freeze_geometry_heads_trains_backbone_only():
    model = VGGTOmega(embed_dim=64, patch_size=16)
    init_model_from_scratch(model)
    trainable, frozen = freeze_geometry_heads(model)
    assert all(not p.requires_grad for p in model.camera_head.parameters())
    assert all(not p.requires_grad for p in model.dense_head.parameters())
    assert any(p.requires_grad for p in model.aggregator.parameters())
    assert trainable > 0 and frozen > 0


from vggt_omega.training.selfsup import OffGridLossComputer

OFFGRID_KEYS = (
    "l1", "l2", "ssim", "lpips",
    "photo_context", "photo_target", "camera", "depth", "render_depth", "total",
)
PHOTO_ONLY = {"camera_weight": 0.0, "depth_weight": 0.0}
#: LossWeights fills unspecified terms with the 3DGS default blend (ssim 0.2), so
#: a test asserting a bare L1 value has to zero the rest explicitly.
L1_ONLY = {"l1": 1.0, "l2": 0.0, "ssim": 0.0, "lpips": 0.0}


def _views(b=1, s=2, hw=16, **kw):
    return torch.rand(b, s, 3, hw, hw, **kw)


def test_offgrid_reports_every_term_even_when_disabled():
    lc = OffGridLossComputer({"l1": 1.0, "l2": 0.0, "ssim": 0.0, "lpips": 0.0})
    out = lc(context=(_views(2, 3), _views(2, 3)))
    assert set(out) == set(OFFGRID_KEYS)
    # Disabled terms are exactly zero, so TensorBoard logging never KeyErrors.
    assert float(out["l2"]) == 0.0 and float(out["ssim"]) == 0.0 and float(out["lpips"]) == 0.0
    # Nothing to distill and nothing held out: those terms report zero, not absent.
    assert float(out["photo_target"]) == 0.0 and float(out["camera"]) == float(out["depth"]) == 0.0


def test_offgrid_l1_only_matches_the_mean_absolute_error():
    lc = OffGridLossComputer({"l1": 1.0, "l2": 0.0, "ssim": 0.0, "lpips": 0.0})
    rendered, target = _views(), _views()
    out = lc(context=(rendered, target))
    torch.testing.assert_close(out["l1"], (rendered - target).abs().mean())
    torch.testing.assert_close(out["total"], out["l1"])


def test_offgrid_perfect_render_is_zero_loss():
    lc = OffGridLossComputer({"l1": 0.8, "l2": 0.0, "ssim": 0.2, "lpips": 0.0})
    target = _views()
    out = lc(context=(target.clone(), target))
    assert out["total"].abs().item() < 1e-5


def test_offgrid_total_is_the_weighted_sum_of_the_reported_terms():
    lc = OffGridLossComputer({"l1": 0.8, "l2": 0.0, "ssim": 0.2, "lpips": 0.0})
    out = lc(context=(_views(), _views()))
    torch.testing.assert_close(out["total"], 0.8 * out["l1"] + 0.2 * out["ssim"])


def test_offgrid_loss_is_differentiable_wrt_the_render():
    lc = OffGridLossComputer({"l1": 0.8, "l2": 0.0, "ssim": 0.2, "lpips": 0.0})
    rendered = _views(requires_grad=True)
    lc(context=(rendered, _views()))["total"].backward()
    assert rendered.grad is not None and torch.isfinite(rendered.grad).all()


def test_offgrid_target_views_add_a_weighted_term():
    lc = OffGridLossComputer(L1_ONLY, target_weight=0.5, **PHOTO_ONLY)
    ctx, tgt = (_views(), _views()), (_views(), _views())
    out = lc(context=ctx, target=(*tgt, torch.ones(1, dtype=torch.bool)))
    torch.testing.assert_close(out["photo_target"], (tgt[0] - tgt[1]).abs().mean())
    torch.testing.assert_close(out["total"], out["photo_context"] + 0.5 * out["photo_target"])


def test_offgrid_refused_items_are_dropped_from_the_target_term():
    """A refused item must not reach the loss at all -- not be averaged in as a
    small number, which would still ask the student to explain a misposed render."""
    lc = OffGridLossComputer(L1_ONLY, **PHOTO_ONLY)
    rendered, images = _views(b=2), _views(b=2)
    kept_only = lc(context=(_views(b=2), _views(b=2)),
                   target=(rendered, images, torch.tensor([True, False])))
    expected = (rendered[:1] - images[:1]).abs().mean()
    torch.testing.assert_close(kept_only["photo_target"], expected)


def test_offgrid_all_refused_is_a_context_only_step():
    lc = OffGridLossComputer({"l1": 1.0}, **PHOTO_ONLY)
    ctx = (_views(), _views())
    out = lc(context=ctx, target=(_views(), _views(), torch.zeros(1, dtype=torch.bool)))
    assert float(out["photo_target"]) == 0.0
    torch.testing.assert_close(out["total"], out["photo_context"])


def test_offgrid_anchor_terms_regress_the_student_onto_the_teacher():
    lc = OffGridLossComputer({"l1": 1.0}, camera_weight=2.0, depth_weight=3.0,
                             conf_weighted_depth=False)
    student = {"pose_enc": torch.zeros(1, 2, 9), "depth": torch.zeros(1, 2, 4, 4, 1)}
    teacher = {"pose_enc": torch.full((1, 2, 9), 0.5), "depth": torch.ones(1, 2, 4, 4, 1)}
    out = lc(context=(_views(), _views()), distill=(student, teacher))
    torch.testing.assert_close(out["camera"], torch.tensor(4.5))  # 9 components x 0.5
    torch.testing.assert_close(out["depth"], torch.tensor(1.0))
    torch.testing.assert_close(out["total"], out["photo_context"] + 2 * 4.5 + 3 * 1.0)


def test_offgrid_anchor_gradient_reaches_the_student_predictions():
    lc = OffGridLossComputer({"l1": 1.0})
    student = {"pose_enc": torch.zeros(1, 2, 9, requires_grad=True),
               "depth": torch.zeros(1, 2, 4, 4, 1, requires_grad=True)}
    teacher = {"pose_enc": torch.full((1, 2, 9), 0.5), "depth": torch.ones(1, 2, 4, 4, 1)}
    lc(context=(_views(), _views()), distill=(student, teacher))["total"].backward()
    assert student["pose_enc"].grad is not None and bool((student["pose_enc"].grad != 0).any())
    assert student["depth"].grad is not None and bool((student["depth"].grad != 0).any())


def test_offgrid_distills_flag_tracks_the_anchor_weights():
    assert OffGridLossComputer({"l1": 1.0}).distills
    assert not OffGridLossComputer({"l1": 1.0}, **PHOTO_ONLY).distills
    assert OffGridLossComputer({"l1": 1.0}, camera_weight=0.0, depth_weight=1.0).distills


def test_offgrid_rejects_an_all_zero_weight_set_at_build_time():
    with pytest.raises(ValueError, match="weight"):
        OffGridLossComputer({"l1": 0.0, "l2": 0.0, "ssim": 0.0, "lpips": 0.0})


def test_offgrid_rejects_an_unknown_weight_key():
    with pytest.raises(ValueError, match="unknown"):
        OffGridLossComputer({"l1": 1.0, "perceptual": 0.5})


from vggt_omega.training.selfsup import (
    align_teacher_trajectory,
    sim3_refuse_gate,
    split_context_target,
)
from vggt_omega.utils.geometry import apply_sim3_to_extrinsics


def test_split_keeps_frame_zero_in_context_and_partitions_all_views():
    ctx, tgt = split_context_target(8, 3, generator=torch.Generator().manual_seed(0))
    assert ctx[0].item() == 0
    assert len(tgt) == 3 and len(ctx) == 5
    assert set(ctx.tolist()) | set(tgt.tolist()) == set(range(8))
    assert not (set(ctx.tolist()) & set(tgt.tolist()))
    assert torch.equal(ctx, ctx.sort().values) and torch.equal(tgt, tgt.sort().values)


def test_split_is_reproducible_for_a_seed_and_varies_across_seeds():
    a = split_context_target(10, 4, generator=torch.Generator().manual_seed(7))[1]
    b = split_context_target(10, 4, generator=torch.Generator().manual_seed(7))[1]
    c = split_context_target(10, 4, generator=torch.Generator().manual_seed(8))[1]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_split_rejects_too_few_views():
    with pytest.raises(ValueError, match=r"n_target \+ 2"):
        split_context_target(4, 3, generator=torch.Generator().manual_seed(0))


def test_split_rejects_non_positive_n_target():
    with pytest.raises(ValueError, match="n_target"):
        split_context_target(8, 0, generator=torch.Generator().manual_seed(0))


def _traj(num_views, seed):
    g = torch.Generator().manual_seed(seed)
    axis = torch.randn(num_views, 3, generator=g)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = torch.rand(num_views, generator=g) * 0.3
    kx, ky, kz = axis.unbind(-1)
    zero = torch.zeros_like(kx)
    K = torch.stack([zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1).reshape(num_views, 3, 3)
    R = torch.eye(3) + torch.sin(angle)[:, None, None] * K + (1 - torch.cos(angle))[:, None, None] * (K @ K)
    return torch.cat([R, torch.randn(num_views, 3, generator=g)[..., None]], dim=-1)


def test_align_teacher_trajectory_recovers_the_student_poses_on_the_overlap():
    # Teacher = an 8-view trajectory whose first 5 poses ARE the student's, all
    # carried through a known Sim(3). Alignment must undo it on the overlap.
    full = _traj(8, seed=1)
    student = full[:5][None]                                # (1, 5, 3, 4)
    rot = _traj(1, seed=4)[0, :3, :3]
    teacher = apply_sim3_to_extrinsics(full, torch.tensor(0.3), rot, torch.tensor([2.0, 0.0, 1.0]))[None]
    ctx = torch.arange(5)

    aligned = align_teacher_trajectory(teacher, student, ctx)
    assert aligned.extrinsics.shape == teacher.shape
    torch.testing.assert_close(aligned.extrinsics[:, ctx], student, atol=1e-4, rtol=1e-4)
    # An exactly recoverable Sim(3) leaves no residual, whatever its magnitude.
    assert float(aligned.residual) < 1e-4
    torch.testing.assert_close(aligned.scale_log, torch.tensor([abs(math.log(0.3))]), atol=1e-4, rtol=1e-4)


def test_align_teacher_trajectory_fits_each_batch_item_independently():
    full_a, full_b = _traj(6, seed=1), _traj(6, seed=2)
    student = torch.stack([full_a[:4], full_b[:4]])
    rot = _traj(1, seed=4)[0, :3, :3]
    teacher = torch.stack([
        apply_sim3_to_extrinsics(full_a, torch.tensor(0.3), rot, torch.tensor([2.0, 0.0, 1.0])),
        apply_sim3_to_extrinsics(full_b, torch.tensor(4.0), rot.T, torch.tensor([-1.0, 5.0, 0.0])),
    ])
    aligned = align_teacher_trajectory(teacher, student, torch.arange(4))
    # Two different transforms recovered in one call.
    torch.testing.assert_close(aligned.extrinsics[:, :4], student, atol=1e-4, rtol=1e-4)
    assert aligned.scale_log.shape == (2,) and float(aligned.residual.max()) < 1e-4


def test_align_teacher_trajectory_returns_detached_poses():
    student = _traj(4, seed=1)[None].requires_grad_(True)
    teacher = _traj(6, seed=3)[None].requires_grad_(True)
    assert not align_teacher_trajectory(teacher, student, torch.arange(4)).extrinsics.requires_grad


def test_alignment_residual_is_large_when_no_sim3_explains_the_overlap():
    """Unrelated trajectories: the fit returns *something*, and the residual is
    what says not to trust it. Without that, a garbage Sim(3) is indistinguishable
    from a good one at the call site."""
    student, teacher = _traj(5, seed=11)[None], _traj(7, seed=12)[None]
    assert float(align_teacher_trajectory(teacher, student, torch.arange(5)).residual) > 0.1


def test_alignment_residual_is_infinite_when_the_student_has_no_spread():
    """All camera centres coincident: there is no trajectory radius to normalise
    by, so the residual is undefined rather than zero -- and must refuse."""
    student = torch.eye(3, 4).expand(1, 4, 3, 4).contiguous()
    aligned = align_teacher_trajectory(_traj(6, seed=3)[None], student, torch.arange(4))
    assert not bool(aligned.residual.isfinite())
    assert not bool(sim3_refuse_gate(aligned, max_residual=0.15)[0])


def _alignment(scale_log=0.0, rotation_deg=0.0, residual=0.0):
    from vggt_omega.training.selfsup import TrajectoryAlignment

    tensor = lambda v: torch.tensor([float(v)])
    return TrajectoryAlignment(
        torch.zeros(1, 4, 3, 4), tensor(scale_log), tensor(rotation_deg), tensor(residual)
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, True),
        ({"scale_log": 0.6}, False),
        ({"rotation_deg": 25.0}, False),
        ({"residual": 0.2}, False),
        ({"scale_log": 0.4, "rotation_deg": 19.0, "residual": 0.14}, True),
    ],
)
def test_refuse_gate_rejects_when_any_single_measure_is_over_its_limit(kwargs, expected):
    keep = sim3_refuse_gate(
        _alignment(**kwargs), max_scale_log=0.5, max_rotation_deg=20.0, max_residual=0.15
    )
    assert bool(keep[0]) is expected


def test_refuse_gate_skips_unset_thresholds():
    wild = _alignment(scale_log=99.0, rotation_deg=179.0, residual=99.0)
    assert bool(sim3_refuse_gate(wild)[0])
    assert bool(sim3_refuse_gate(wild, max_scale_log=None, max_residual=float("inf"))[0])


def test_refuse_gate_is_per_batch_item():
    from vggt_omega.training.selfsup import TrajectoryAlignment

    zeros = torch.zeros(3)
    alignment = TrajectoryAlignment(
        torch.zeros(3, 4, 3, 4), zeros, zeros, torch.tensor([0.01, 0.9, 0.02])
    )
    assert sim3_refuse_gate(alignment, max_residual=0.15).tolist() == [True, False, True]


# --- rendered-depth anchor ----------------------------------------------------
def _rd(weight=1.0, min_alpha=0.5):
    return OffGridLossComputer(L1_ONLY, render_depth_weight=weight,
                               render_depth_min_alpha=min_alpha, **PHOTO_ONLY)


def test_render_depth_is_l1_where_the_render_has_cover():
    lc = _rd()
    depth = torch.full((1, 2, 4, 4), 3.0)
    alpha = torch.ones(1, 2, 4, 4)
    teacher = torch.full((1, 2, 4, 4, 1), 5.0)
    torch.testing.assert_close(lc.render_depth_loss(depth, alpha, teacher), torch.tensor(2.0))


def test_render_depth_ignores_uncovered_pixels():
    """gsplat's depth is an alpha-weighted sum, so an uncovered pixel is 0 with 0
    alpha -- UNDEFINED, not 'far'. Regressing it toward the teacher's real depth
    would train the model to explain empty space instead of filling it."""
    lc = _rd()
    depth = torch.full((1, 1, 2, 2), 3.0)
    alpha = torch.ones(1, 1, 2, 2)
    depth[0, 0, 0, 0] = 0.0          # a hole: wildly wrong if it were scored
    alpha[0, 0, 0, 0] = 0.0
    teacher = torch.full((1, 1, 2, 2, 1), 4.0)
    # Only the 3 covered pixels count, each off by 1.0.
    torch.testing.assert_close(lc.render_depth_loss(depth, alpha, teacher), torch.tensor(1.0))


def test_render_depth_is_zero_when_nothing_is_covered():
    """No cover anywhere must not divide by zero or return NaN."""
    lc = _rd()
    out = lc.render_depth_loss(
        torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), torch.full((1, 1, 2, 2, 1), 4.0)
    )
    assert torch.isfinite(out) and float(out) == 0.0


def test_render_depth_threshold_selects_which_pixels_count():
    lc_low, lc_high = _rd(min_alpha=0.1), _rd(min_alpha=0.9)
    depth = torch.tensor([[[[3.0, 3.0]]]])
    alpha = torch.tensor([[[[0.95, 0.5]]]])       # one confident, one marginal
    teacher = torch.tensor([[[[[4.0], [10.0]]]]])
    # min_alpha 0.9 keeps only the confident pixel (err 1.0); 0.1 keeps both.
    torch.testing.assert_close(lc_high.render_depth_loss(depth, alpha, teacher), torch.tensor(1.0))
    torch.testing.assert_close(lc_low.render_depth_loss(depth, alpha, teacher), torch.tensor(4.0))


def test_render_depth_enters_the_total_and_is_reported():
    lc = _rd(weight=3.0)
    views = (_views(), _views())
    rd = (torch.full((1, 2, 4, 4), 3.0), torch.ones(1, 2, 4, 4), torch.full((1, 2, 4, 4, 1), 5.0))
    out = lc(context=views, render_depth=rd)
    torch.testing.assert_close(out["render_depth"], torch.tensor(2.0))
    torch.testing.assert_close(out["total"], out["photo_context"] + 3.0 * 2.0)


def test_render_depth_off_by_default_and_reported_as_zero():
    """Existing configs must be untouched -- but the key still has to exist, or
    the trainer's fixed logging tuple KeyErrors."""
    lc = OffGridLossComputer(L1_ONLY, **PHOTO_ONLY)
    assert lc.render_depth_weight == 0.0
    out = lc(context=(_views(), _views()))
    assert float(out["render_depth"]) == 0.0


def test_render_depth_alone_still_requires_the_teacher_forward():
    lc = OffGridLossComputer(L1_ONLY, camera_weight=0.0, depth_weight=0.0, render_depth_weight=1.0)
    assert lc.distills, "render_depth needs the context-view teacher depth"


def test_render_depth_gradient_reaches_the_rendered_depth():
    lc = _rd()
    depth = torch.full((1, 1, 4, 4), 3.0, requires_grad=True)
    lc.render_depth_loss(depth, torch.ones(1, 1, 4, 4), torch.full((1, 1, 4, 4, 1), 5.0)).backward()
    assert depth.grad is not None and bool((depth.grad != 0).any())


def test_alignment_refuses_non_finite_trajectory():
    """A non-finite student trajectory must not raise, and must refuse.

    torch.linalg.svd raises _LinAlgError on non-finite input rather than
    returning something the degeneracy checks could reject, which killed three
    2-node runs inside a @torch.no_grad CPU helper. The alignment now falls back
    to identity and reports infinite diagnostics, which sim3_refuse_gate treats
    as a refusal -- so the held-out target views are dropped and the step keeps
    its context supervision.
    """
    from vggt_omega.training.selfsup import align_teacher_trajectory, sim3_refuse_gate

    B, S = 1, 6
    teacher_ext = torch.eye(4)[:3].repeat(B, S, 1, 1).clone()
    teacher_ext[:, :, 0, 3] = torch.arange(S, dtype=torch.float32)  # spread the centres
    student_ext = teacher_ext.clone()
    student_ext[0, 2, 0, 3] = float("nan")  # one bad camera poisons the fit
    context_idx = torch.arange(S)

    alignment = align_teacher_trajectory(teacher_ext, student_ext, context_idx)

    assert torch.isfinite(alignment.extrinsics).all(), "aligned poses reach the rasterizer"
    keep = sim3_refuse_gate(
        alignment, max_scale_log=0.25, max_rotation_deg=15.0, max_residual=0.15
    )
    assert not bool(keep.any()), "a non-finite alignment must set the gate to 0"


def test_umeyama_does_not_raise_on_non_finite():
    """The guard also sits in umeyama_sim3 itself, for any other caller."""
    from vggt_omega.utils.geometry import umeyama_sim3

    src = torch.randn(6, 3)
    dst = src.clone()
    dst[1, 0] = float("inf")
    scale, rotation, translation = umeyama_sim3(src, dst)  # must not raise
    assert rotation.shape == (3, 3)
