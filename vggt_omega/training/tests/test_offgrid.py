"""CPU unit tests for the off-the-grid (M context + N target) training path.

These exercise Trainer methods against lightweight stand-ins rather than a real
Trainer: building one constructs a model and a process group, which needs a GPU
and would make a unit test a smoke test. verify/offgrid_smoke.py covers the
wired-up loop end to end.
"""
import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from vggt_omega.training.trainer import Trainer, resolve_selfsup_mode


class _RecordingOptimizer:
    def __init__(self):
        self.steps = 0
        self.zeroed = 0

    def step(self):
        self.steps += 1

    def zero_grad(self, set_to_none=False):
        self.zeroed += 1


class _RecordingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def _fake_trainer(ema_teacher=None):
    return SimpleNamespace(
        optimizer=_RecordingOptimizer(),
        scheduler=_RecordingScheduler(),
        ema_teacher=ema_teacher,
        model=object(),
        rank=0,
        global_step=7,
        nonfinite_grad_steps=0,
    )


def test_step_optimizer_steps_on_a_finite_grad_norm():
    t = _fake_trainer()
    Trainer._step_optimizer(t, 1.25)
    assert t.optimizer.steps == 1
    assert t.nonfinite_grad_steps == 0
    assert t.optimizer.zeroed == 1 and t.scheduler.steps == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_step_optimizer_skips_the_update_on_a_nonfinite_grad_norm(bad):
    t = _fake_trainer()
    Trainer._step_optimizer(t, bad)
    assert t.optimizer.steps == 0
    assert t.nonfinite_grad_steps == 1
    # The schedule is a function of step count, so it still advances; gradients
    # are still cleared so the bad batch cannot leak into the next step.
    assert t.optimizer.zeroed == 1 and t.scheduler.steps == 1


def test_step_optimizer_skips_the_ema_update_on_a_nonfinite_grad_norm():
    updates = []
    ema = SimpleNamespace(update=lambda model, step: updates.append(step))
    Trainer._step_optimizer(_fake_trainer(ema_teacher=ema), float("nan"))
    assert updates == []
    Trainer._step_optimizer(_fake_trainer(ema_teacher=ema), 0.5)
    assert updates == [7]


def test_resolve_selfsup_mode_defaults_to_feature():
    assert resolve_selfsup_mode(OmegaConf.create({"selfsup": {"enabled": True}})) == "feature"
    assert resolve_selfsup_mode(OmegaConf.create({})) == "feature"


def test_resolve_selfsup_mode_rejects_an_unknown_mode():
    cfg = OmegaConf.create({"selfsup": {"enabled": True, "mode": "nonsense"}})
    with pytest.raises(ValueError, match="selfsup.mode"):
        resolve_selfsup_mode(cfg)


def _offgrid_cfg(**overrides):
    cfg = OmegaConf.create({
        "selfsup": {"enabled": True, "mode": "offgrid", "n_target": 2,
                    "weights": {"l1": 0.8, "l2": 0.0, "ssim": 0.2, "lpips": 0.0}},
        "model": {"enable_3dgs": True},
    })
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_build_offgrid_loss_requires_a_teacher():
    fake = SimpleNamespace(cfg=_offgrid_cfg(), teacher=None, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="teacher"):
        Trainer._build_offgrid_loss(fake)


def test_build_offgrid_loss_requires_the_gaussian_head():
    fake = SimpleNamespace(
        cfg=_offgrid_cfg(model={"enable_3dgs": False}),
        teacher=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="enable_3dgs"):
        Trainer._build_offgrid_loss(fake)


def test_build_offgrid_loss_returns_a_photometric_computer():
    fake = SimpleNamespace(cfg=_offgrid_cfg(), teacher=object(), device=torch.device("cpu"))
    views = (torch.rand(1, 2, 3, 16, 16), torch.rand(1, 2, 3, 16, 16))
    out = Trainer._build_offgrid_loss(fake)(context=views)
    assert set(out) == {"l1", "l2", "ssim", "lpips", "photo_context", "photo_target",
                        "camera", "depth", "render_depth", "total"}


def test_build_offgrid_loss_defaults_the_weights_when_unset():
    cfg = _offgrid_cfg()
    del cfg.selfsup.weights
    fake = SimpleNamespace(cfg=cfg, teacher=object(), device=torch.device("cpu"))
    weights = Trainer._build_offgrid_loss(fake).weights
    assert (weights.l1, weights.l2, weights.ssim, weights.lpips) == (0.8, 0.0, 0.2, 0.0)


def test_build_offgrid_loss_anchors_to_the_teacher_by_default():
    """Photometric-only is the configuration that collapses the fov, so the
    teacher anchor has to be what you get for free, not what you remember to ask for."""
    fake = SimpleNamespace(cfg=_offgrid_cfg(), teacher=object(), device=torch.device("cpu"))
    assert Trainer._build_offgrid_loss(fake).distills


def test_build_offgrid_loss_reads_the_gate_thresholds():
    fake = SimpleNamespace(
        cfg=_offgrid_cfg(selfsup={"gate": {"max_residual": 0.05}, "target_weight": 0.5}),
        teacher=object(),
        device=torch.device("cpu"),
    )
    computer = Trainer._build_offgrid_loss(fake)
    assert computer.target_weight == 0.5
    assert fake.gate_cfg["max_residual"] == 0.05
    # Unset thresholds keep their defaults rather than becoming None (= no gate).
    assert fake.gate_cfg["max_scale_log"] == 0.5 and fake.gate_cfg["max_rotation_deg"] == 20.0


def test_alignment_diagnostics_replace_a_degenerate_fit_with_a_finite_sentinel():
    from vggt_omega.training.selfsup import TrajectoryAlignment

    inf = torch.tensor([float("inf")])
    alignment = TrajectoryAlignment(torch.zeros(1, 2, 3, 4), inf, torch.tensor([3.0]), inf)
    out = Trainer._alignment_diagnostics(alignment, torch.zeros(1, dtype=torch.bool))
    assert float(out["gate_rate"]) == 0.0
    assert all(torch.isfinite(v).all() for v in out.values())
    assert float(out["sim3_rotation_deg"]) == 3.0


# --- degenerate splat guard ---------------------------------------------------
def _gaussians(n=6, d_sh=4):
    from vggt_omega.models.heads.gaussian_head import Gaussians

    return Gaussians(
        means=torch.zeros(1, n, 3), covariances=torch.zeros(1, n, 3, 3),
        harmonics=torch.zeros(1, n, 3, d_sh), opacities=torch.full((1, n), 0.5),
        scales=torch.full((1, n, 3), 0.01), rotations=torch.zeros(1, n, 4),
    )


def test_sanitize_is_a_no_op_when_every_splat_is_finite():
    from vggt_omega.training.selfsup import sanitize_gaussians

    g = _gaussians()
    out, count = sanitize_gaussians(g)
    assert count == 0 and out is g  # identity, not a copy: the hot path pays nothing


def test_sanitize_zeroes_the_opacity_of_a_splat_with_a_non_finite_mean():
    """A NaN mean is what crashes gsplat's backward on MUSA; the splat has to
    stop contributing, not merely have its coordinate patched to the origin."""
    from vggt_omega.training.selfsup import sanitize_gaussians

    g = _gaussians()
    g.means[0, 2, 1] = float("nan")
    out, count = sanitize_gaussians(g)
    assert count == 1
    assert float(out.opacities[0, 2]) == 0.0
    assert float(out.opacities[0, 0]) == 0.5  # neighbours untouched
    assert torch.isfinite(out.means).all()


def test_sanitize_catches_every_guarded_field():
    from vggt_omega.training.selfsup import sanitize_gaussians

    for field, index in (("scales", (0, 1, 0)), ("rotations", (0, 3, 2)),
                         ("harmonics", (0, 4, 1, 0)), ("opacities", (0, 5))):
        g = _gaussians()
        getattr(g, field)[index] = float("inf")
        out, count = sanitize_gaussians(g)
        assert count == 1, f"{field} not guarded"
        assert float(out.opacities[0, index[1]]) == 0.0
        assert torch.isfinite(getattr(out, field)).all()


def test_sanitize_keeps_gradients_flowing_for_the_good_splats():
    from vggt_omega.training.selfsup import sanitize_gaussians

    g = _gaussians()
    g.means = g.means.clone().requires_grad_(True)
    bad = g.scales.clone()
    bad[0, 1] = float("nan")
    g.scales = bad
    out, count = sanitize_gaussians(g)
    assert count == 1
    out.means.sum().backward()
    assert g.means.grad is not None and torch.isfinite(g.means.grad).all()


# --- gradient accumulation ----------------------------------------------------
class _Recorder:
    """Stands in for the pieces _finish_micro_step drives, to record WHEN they fire."""

    def __init__(self, accumulate, clip=1.0):
        self.accumulate_steps = accumulate
        self.cp_group = None
        self.model = torch.nn.Linear(2, 2)
        self.cfg = OmegaConf.create({"optim": {"grad_clip": clip}})
        self.updates = []
        self.rank = 0

    def _step_optimizer(self, grad_norm):
        self.updates.append(grad_norm)


def _run_micro_steps(recorder, count):
    """Feed `count` micro-batches through, closing every accumulate_steps-th."""
    grads = []
    for i in range(count):
        loss = (recorder.model(torch.ones(1, 2)) ** 2).sum()
        apply_update = (i + 1) % recorder.accumulate_steps == 0
        grads.append(Trainer._finish_micro_step(recorder, loss, apply_update))
    return grads


def test_accumulation_defers_the_optimizer_to_the_boundary():
    r = _Recorder(accumulate=4)
    _run_micro_steps(r, 8)
    # 8 micro-batches at N=4 -> exactly 2 optimizer steps. (The sibling repo
    # also asserts its CP gradient reduction fires here; CP is not ported.)
    assert len(r.updates) == 2, f"expected 2 updates, got {len(r.updates)}"


def test_non_boundary_micro_steps_report_nan_not_zero():
    """A 0.0 grad norm would read as a real measurement in a log; NaN cannot."""
    r = _Recorder(accumulate=3)
    grads = _run_micro_steps(r, 3)
    assert math.isnan(grads[0]) and math.isnan(grads[1])
    assert not math.isnan(grads[2]), "the boundary step must report a real norm"


def test_accumulate_one_is_exactly_the_old_behaviour():
    r = _Recorder(accumulate=1)
    _run_micro_steps(r, 5)
    assert len(r.updates) == 5, "N=1 must update on every micro-batch"


def test_accumulated_gradient_equals_the_mean_over_micro_batches():
    """The whole point: N micro-batches must give the gradient of ONE batch of
    that size, not N times it. Dividing the loss by N is what makes that true.

    Every micro-step here is a NON-boundary one, which isolates the accumulation
    from the update path: clip_grad_norm_ rescales .grad IN PLACE at the boundary
    and would rewrite the very quantity being compared.
    """
    torch.manual_seed(0)
    inputs = [torch.randn(1, 2) for _ in range(4)]
    r = _Recorder(accumulate=4)
    model = r.model

    # Reference: one batch, loss averaged over the four samples.
    torch.stack([(model(x) ** 2).sum() for x in inputs]).mean().backward()
    want = model.weight.grad.clone()
    model.zero_grad(set_to_none=True)

    for x in inputs:
        Trainer._finish_micro_step(r, (model(x) ** 2).sum(), apply_update=False)
    assert not r.updates, "no boundary was crossed, so nothing should have stepped"
    torch.testing.assert_close(model.weight.grad, want)
