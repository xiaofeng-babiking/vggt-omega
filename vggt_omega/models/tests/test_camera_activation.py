# vggt_omega/models/tests/test_camera_activation.py
"""The fov activation must be escapable, and a no-op on the real operating range.

``fov = F.relu(raw) + 0.01`` is an ABSORBING state: once raw goes negative the
gradient through the relu is exactly zero, so nothing -- neither the photometric
term nor the teacher anchor -- can ever push the fov back up. A camera that falls
in is a pencil camera forever (fy = (H/2)/tan(0.005) ~= 51200 at H=512), which is
the documented "memorised dust" collapse.

The replacement has to satisfy two things at once, and the second is what makes
it safe to ship against pretrained weights:

  escapable  -- non-zero gradient everywhere, including deep in the dead zone.
  identity   -- indistinguishable from relu wherever real cameras actually live.
                verify/fov_range_probe.py measured the pretrained checkpoint over
                garden/room/bicycle/counter: fov in [0.61, 1.21] rad, i.e. 61x the
                0.01 floor, with ~0.6 of raw margin. The teacher shares this code
                path, so an activation that shifted predictions there would move
                the anchor's own target and corrupt the oracle.
"""
import pytest
import torch

from vggt_omega.models.heads.camera_head import _apply_camera_activation

# The measured pretrained range (verify/fov_range_probe.py), plus headroom.
OPERATING_RANGE = (0.55, 1.30)


def _fov(raw_fov, **kw):
    """Run the activation on a (..., 9) pose vector, return just the fov pair."""
    raw = torch.zeros(raw_fov.shape[:-1] + (9,), dtype=raw_fov.dtype)
    raw[..., 7:] = raw_fov
    return _apply_camera_activation(raw, **kw)[..., 7:]


def test_softplus_matches_relu_on_the_operating_range():
    """Pretrained compatibility: on real fovs the new activation IS the old one."""
    raw = torch.linspace(*OPERATING_RANGE, 64).reshape(-1, 1).repeat(1, 2)
    got = _fov(raw, fov_activation="softplus")
    relu = _fov(raw, fov_activation="relu")
    assert torch.allclose(got, relu, atol=1e-6), (got - relu).abs().max()


def test_relu_mode_is_bit_exact_with_the_original():
    """The old behaviour stays reachable for reproducing existing checkpoints."""
    raw = torch.linspace(-2.0, 2.0, 64).reshape(-1, 1).repeat(1, 2)
    got = _fov(raw, fov_activation="relu")
    expected = torch.relu(raw) + 0.01
    assert torch.equal(got, expected)


def test_relu_dead_zone_has_exactly_zero_gradient():
    """Characterises the bug being fixed -- if this ever fails, the trap is gone."""
    raw = torch.full((1, 2), -0.5, requires_grad=True)
    _fov(raw, fov_activation="relu").sum().backward()
    assert torch.equal(raw.grad, torch.zeros_like(raw.grad))


@pytest.mark.parametrize("raw_value", [-0.05, -0.5, -2.0, -8.0])
def test_softplus_keeps_a_live_gradient_in_the_dead_zone(raw_value):
    """The fix: a camera that fell in can still climb back out."""
    raw = torch.full((1, 2), raw_value, dtype=torch.float64, requires_grad=True)
    _fov(raw, fov_activation="softplus").sum().backward()
    assert (raw.grad > 0).all(), f"gradient died at raw={raw_value}: {raw.grad}"


def test_gradient_is_usable_where_a_camera_would_first_fall_in():
    """Escapability is only worth anything if the gradient is big enough to act on.

    A camera does not teleport to raw=-8; it crosses zero. Just inside the dead
    zone the derivative is sigmoid(beta*raw), which at raw=-0.05 is ~0.076 -- the
    same order as the healthy region, so the anchor can pull it straight back.
    Deeper in, the gradient decays exponentially: this catches a camera as it
    falls rather than resurrecting one already lost, which is the failure path
    that actually occurs.
    """
    raw = torch.full((1, 2), -0.05, dtype=torch.float64, requires_grad=True)
    _fov(raw, fov_activation="softplus").sum().backward()
    assert (raw.grad > 1e-2).all(), raw.grad


@pytest.mark.parametrize("raw_value", [-50.0, -8.0, -0.5, 0.0, 0.5, 3.0])
def test_fov_stays_strictly_positive(raw_value):
    """tan(fov/2) is the focal denominator -- a non-positive fov is a divide-by-zero."""
    raw = torch.full((1, 2), raw_value, dtype=torch.float64)
    assert (_fov(raw, fov_activation="softplus") > 0).all()


def test_fov_is_non_decreasing_everywhere():
    """Ordering must survive the swap, or the head's learned ranking is scrambled."""
    raw = torch.linspace(-3.0, 3.0, 128, dtype=torch.float64).reshape(-1, 1).repeat(1, 2)
    fov = _fov(raw, fov_activation="softplus")
    assert (fov.diff(dim=0) >= 0).all()


def test_fov_is_strictly_increasing_where_it_is_representable():
    """Deep in the dead zone the softplus falls below the ULP of FOV_MIN, so
    ``FOV_MIN + tiny`` rounds back to FOV_MIN and neighbouring values tie. That is
    a float-resolution artefact of having a positive floor at all -- relu has the
    same flat value there AND no gradient. What matters is that the ordering is
    strict wherever the value is representable, which covers the whole approach to
    the floor."""
    raw = torch.linspace(-0.5, 3.0, 128, dtype=torch.float64).reshape(-1, 1).repeat(1, 2)
    fov = _fov(raw, fov_activation="softplus")
    assert (fov.diff(dim=0) > 0).all()


def test_translation_and_quaternion_are_untouched():
    """Only components 7:9 are activated; the rest must pass through verbatim."""
    raw = torch.randn(4, 9, dtype=torch.float64)
    out = _apply_camera_activation(raw, fov_activation="softplus")
    assert torch.equal(out[..., :7], raw[..., :7])


def test_rejects_unknown_activation():
    with pytest.raises(ValueError, match="fov_activation"):
        _apply_camera_activation(torch.zeros(1, 9), fov_activation="gelu")


def test_default_is_the_escapable_activation():
    """A run that configures nothing must get the fixed behaviour, not the trap."""
    raw = torch.full((1, 2), -0.5, dtype=torch.float64, requires_grad=True)
    _fov(raw).sum().backward()
    assert (raw.grad > 0).all()


def test_does_not_use_the_fused_softplus_kernel():
    """Regression guard for a MUSA-only reinstatement of the trapdoor.

    ``F.softplus`` silently returns exactly 0.0 on MUSA for inputs <= -16.636
    (gaussian_head.stable_softplus). At FOV_SOFTPLUS_BETA=50 that is
    raw <= -0.333 -- comfortably inside the region this activation exists to keep
    alive, and invisible to every CPU test in this file. The activation must go
    through the relu/exp/log1p identity instead, so assert the call is absent
    rather than trusting the comment above it.
    """
    import inspect

    from vggt_omega.models.heads import camera_head

    # Strip comments first: the implementation deliberately NAMES F.softplus in a
    # comment explaining why it is avoided, and matching prose would be a
    # self-defeating test.
    src = inspect.getsource(camera_head._apply_camera_activation)
    code = "\n".join(line.split("#")[0] for line in src.splitlines())
    assert "F.softplus(" not in code, "fused softplus is wrong on MUSA below -16.636/beta"
    assert "stable_softplus(" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_gradient_survives_the_dead_zone_on_device():
    """The same escape property, on the accelerator that actually trains.

    raw=-0.5 puts beta*raw at -25 — past the threshold where the sibling MUSA
    repo's fused kernel zeroed out. CUDA has no such defect, but the property
    this file exists for (a live gradient below zero) must hold on device, not
    just in the CPU tests above.
    """
    raw = torch.full((1, 2), -0.5, device="cuda", requires_grad=True)
    _fov(raw, fov_activation="softplus").sum().backward()
    assert (raw.grad > 0).all(), raw.grad
