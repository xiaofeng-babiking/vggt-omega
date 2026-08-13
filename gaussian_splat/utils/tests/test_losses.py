"""Tests for the photometric / perceptual losses.

Everything but the fused-SSIM parity check runs on CPU. LPIPS tests use a
*random* trunk (``pretrained_trunk=False``) so they never touch the network;
they check plumbing (shape, range, gradients, chunking), not the metric.
"""

import pytest
import torch

_HAS_CUDA = torch.cuda.is_available()

from gaussian_splat.utils.losses import (
    LossWeights,
    l1_loss,
    l2_loss,
    lpips_loss,
    photometric_loss,
    ssim,
    ssim_loss,
)

_B, _V, _H, _W = 2, 3, 32, 40


def _image_pair(seed: int = 0, shape=(_B, _V, 3, _H, _W)):
    generator = torch.Generator().manual_seed(seed)
    pred = torch.rand(shape, generator=generator)
    target = torch.rand(shape, generator=generator)
    return pred, target


# --------------------------------------------------------------------------
# l1 / l2
# --------------------------------------------------------------------------


def test_per_pixel_losses_are_zero_for_identical_images():
    pred, _ = _image_pair()
    assert l1_loss(pred, pred.clone()) == 0.0
    assert l2_loss(pred, pred.clone()) == 0.0


def test_per_pixel_losses_match_closed_form():
    pred = torch.full((1, 3, 4, 4), 0.75)
    target = torch.full((1, 3, 4, 4), 0.25)
    assert l1_loss(pred, target).item() == pytest.approx(0.5)
    assert l2_loss(pred, target).item() == pytest.approx(0.25)


def test_reductions():
    pred, target = _image_pair()
    per_element = l1_loss(pred, target, reduction="none")
    assert per_element.shape == pred.shape
    assert l1_loss(pred, target, reduction="sum").item() == pytest.approx(per_element.sum().item())
    assert l1_loss(pred, target).item() == pytest.approx(per_element.mean().item(), rel=1e-6)


def test_mask_averages_over_kept_pixels_only():
    pred, target = _image_pair()
    mask = torch.zeros(_B, _V, _H, _W, dtype=torch.bool)
    mask[..., :8, :] = True  # keep the top 8 rows

    masked = l1_loss(pred, target, mask=mask)
    cropped = l1_loss(pred[..., :8, :], target[..., :8, :])
    assert masked.item() == pytest.approx(cropped.item(), rel=1e-6)


def test_mask_accepts_channel_axis_and_float_weights():
    pred, target = _image_pair()
    mask_hw = torch.zeros(_B, _V, _H, _W)
    mask_hw[..., :8, :] = 1.0
    mask_chw = mask_hw.unsqueeze(-3).expand(_B, _V, 3, _H, _W)
    assert l1_loss(pred, target, mask=mask_hw).item() == pytest.approx(
        l1_loss(pred, target, mask=mask_chw).item(), rel=1e-6
    )


def test_fully_masked_batch_is_zero_not_nan():
    pred, target = _image_pair()
    mask = torch.zeros(_B, _V, _H, _W, dtype=torch.bool)
    loss = l1_loss(pred, target, mask=mask)
    assert loss.item() == 0.0
    assert torch.isfinite(loss)


def test_misaligned_mask_raises():
    pred, target = _image_pair()
    with pytest.raises(ValueError, match="does not align"):
        l1_loss(pred, target, mask=torch.ones(_H, _W))


# --------------------------------------------------------------------------
# ssim
# --------------------------------------------------------------------------


def test_ssim_of_identical_images_is_one():
    pred, _ = _image_pair()
    assert ssim(pred, pred.clone(), backend="torch").item() == pytest.approx(1.0, abs=1e-5)
    assert ssim_loss(pred, pred.clone(), backend="torch").item() == pytest.approx(0.0, abs=1e-5)


def test_ssim_degrades_with_noise():
    pred, _ = _image_pair()
    noisy = (pred + 0.2 * torch.randn_like(pred)).clamp(0.0, 1.0)
    clean_score = ssim(pred, pred.clone(), backend="torch")
    noisy_score = ssim(noisy, pred, backend="torch")
    assert noisy_score < clean_score
    assert -1.0 <= noisy_score.item() <= 1.0


def test_ssim_valid_padding_drops_the_border():
    pred, target = _image_pair()
    same = ssim(pred, target, padding="same", backend="torch")
    valid = ssim(pred, target, padding="valid", backend="torch")
    # Border windows see zeros outside the image, so the two disagree; both
    # must stay in range and finite.
    assert same.item() != pytest.approx(valid.item(), abs=1e-6)
    assert torch.isfinite(valid) and -1.0 <= valid.item() <= 1.0


def test_ssim_flattens_leading_dims():
    pred, target = _image_pair()
    flat = ssim(pred.reshape(_B * _V, 3, _H, _W), target.reshape(_B * _V, 3, _H, _W), backend="torch")
    assert ssim(pred, target, backend="torch").item() == pytest.approx(flat.item(), rel=1e-6)


def test_ssim_backpropagates_into_pred_only():
    pred, target = _image_pair()
    pred = pred.requires_grad_(True)
    target = target.requires_grad_(True)
    ssim_loss(pred, target, backend="torch").backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_ssim_rejects_bad_arguments():
    pred, target = _image_pair()
    with pytest.raises(ValueError, match="padding"):
        ssim(pred, target, padding="reflect")
    with pytest.raises(ValueError, match="backend"):
        ssim(pred, target, backend="opencv")
    with pytest.raises(ValueError, match="shape mismatch"):
        ssim(pred, target[..., :16, :])


def test_ssim_is_unaffected_by_autocast():
    """Autocast must not reach the SSIM math, or it drifts from the kernel."""
    pred, target = _image_pair()
    baseline = ssim(pred, target, backend="torch")
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        inside = ssim(pred, target, backend="torch")
    assert inside.dtype == torch.float32
    assert inside.item() == pytest.approx(baseline.item(), rel=1e-6)


def test_ssim_promotes_bf16_inputs():
    pred, target = _image_pair()
    got = ssim(pred.bfloat16(), target.bfloat16(), backend="torch")
    # Promotion is exactly a bf16 round-trip, so this holds to fp32 precision.
    expected = ssim(pred.bfloat16().float(), target.bfloat16().float(), backend="torch")
    assert got.dtype == torch.float32
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)


def test_ssim_backpropagates_through_a_bf16_input():
    pred, target = _image_pair()
    pred = pred.bfloat16().requires_grad_(True)
    ssim_loss(pred, target.bfloat16(), backend="torch").backward()
    assert pred.grad is not None and pred.grad.dtype == torch.bfloat16
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


@pytest.mark.parametrize("shape", [(1, 3, 4, 4), (1, 3, 10, 10), (1, 3, 64, 8)])
def test_ssim_valid_padding_rejects_images_it_would_empty(shape):
    """An empty crop means NaN on CPU but 0.0 on musa — refuse it instead."""
    x = torch.rand(shape)
    with pytest.raises(ValueError, match="empties"):
        ssim(x, x.clone(), padding="valid", backend="torch")


def test_ssim_valid_padding_allows_the_smallest_workable_image():
    x = torch.rand(1, 3, 11, 11)
    assert ssim(x, x.clone(), padding="valid", backend="torch").item() == pytest.approx(
        1.0, abs=1e-5
    )


@pytest.mark.skipif(not _HAS_CUDA, reason="fused-ssim kernel needs a CUDA device")
def test_fused_ssim_accepts_bf16_and_autocast():
    """Both used to fail: bf16 raised outright, autocast diverged from fp32."""
    pytest.importorskip("fused_ssim")
    pred, target = _image_pair()
    pred, target = pred.to("cuda"), target.to("cuda")

    baseline = ssim(pred, target, backend="fused")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        assert ssim(pred, target, backend="fused").item() == pytest.approx(
            baseline.item(), rel=1e-6
        )

    got = ssim(pred.bfloat16(), target.bfloat16(), backend="fused")
    expected = ssim(pred.bfloat16().float(), target.bfloat16().float(), backend="fused")
    assert got.item() == pytest.approx(expected.item(), rel=1e-6)


@pytest.mark.skipif(not _HAS_CUDA, reason="fused-ssim kernel needs a CUDA device")
def test_fused_and_torch_ssim_agree():
    """The two backends must be interchangeable — same value, same gradient."""
    pytest.importorskip("fused_ssim")
    pred, target = _image_pair()
    pred, target = pred.to("cuda"), target.to("cuda")

    scores, grads = [], []
    for backend in ("torch", "fused"):
        x = pred.clone().requires_grad_(True)
        score = ssim(x, target, backend=backend)
        score.backward()
        scores.append(score.item())
        grads.append(x.grad)

    assert scores[0] == pytest.approx(scores[1], abs=1e-5)
    torch.testing.assert_close(grads[0], grads[1], atol=1e-5, rtol=1e-4)


# --------------------------------------------------------------------------
# lpips
# --------------------------------------------------------------------------


def test_lpips_is_zero_for_identical_images_and_positive_otherwise():
    pytest.importorskip("lpips")
    pred, target = _image_pair(shape=(2, 3, 64, 64))
    same = lpips_loss(pred, pred.clone(), net="alex", pretrained_trunk=False)
    different = lpips_loss(pred, target, net="alex", pretrained_trunk=False)
    assert same.item() == pytest.approx(0.0, abs=1e-6)
    assert different.item() > 0.0
    assert same.shape == torch.Size([])


def test_lpips_chunking_matches_a_single_pass():
    pytest.importorskip("lpips")
    pred, target = _image_pair(shape=(2, 2, 3, 64, 64))
    whole = lpips_loss(pred, target, net="alex", pretrained_trunk=False)
    chunked = lpips_loss(pred, target, net="alex", pretrained_trunk=False, chunk_size=1)
    assert whole.item() == pytest.approx(chunked.item(), rel=1e-5)


def test_lpips_backpropagates_into_pred():
    pytest.importorskip("lpips")
    pred, target = _image_pair(shape=(1, 3, 64, 64))
    pred = pred.requires_grad_(True)
    lpips_loss(pred, target, net="alex", pretrained_trunk=False).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_lpips_rejects_non_rgb():
    pytest.importorskip("lpips")
    pred, target = _image_pair(shape=(1, 1, 64, 64))
    with pytest.raises(ValueError, match="3-channel"):
        lpips_loss(pred, target, pretrained_trunk=False)


@pytest.mark.parametrize(
    "name, make",
    [
        ("out of range", lambda: (torch.rand(2, 3, 64, 64) - 0.5) * 100),
        ("huge", lambda: (torch.rand(2, 3, 64, 64) - 0.5) * 2e4),
        ("has inf", lambda: torch.cat(
            [torch.full((1, 3, 64, 64), float("inf")), torch.rand(1, 3, 64, 64)])),
        ("has nan", lambda: torch.cat(
            [torch.full((1, 3, 64, 64), float("nan")), torch.rand(1, 3, 64, 64)])),
    ],
)
def test_lpips_survives_an_unbounded_render(name, make):
    """A rasterizer output is not bounded; LPIPS is only defined on [0, 1].

    Feeding out-of-range values through the trunk's feature normalisation
    (x / (sqrt(sum(x^2)) + eps)) returns nan, which poisons the whole loss.
    Measured on a 9-scene run before the clamp: 1453 of 9000 steps went
    non-finite, 90% attributed to this term, with l1 and l2 finite on the very
    same renders. The value need not be meaningful for a garbage render -- it
    must be FINITE, and the gradient with it, so one bad sample cannot take
    down a step that seven other ranks are contributing to.
    """
    pytest.importorskip("lpips")
    target = torch.rand(2, 3, 64, 64)
    pred = make().requires_grad_(True)
    value = lpips_loss(pred, target, net="alex", pretrained_trunk=False)
    value.backward()
    assert torch.isfinite(value), f"{name}: loss is {float(value)}"
    assert torch.isfinite(pred.grad).all(), f"{name}: non-finite gradient"


# --------------------------------------------------------------------------
# combination
# --------------------------------------------------------------------------


def test_photometric_loss_only_evaluates_weighted_terms():
    pred, target = _image_pair()
    total, terms = photometric_loss(
        pred, target, weights=LossWeights(l1=1.0, l2=0.0, ssim=0.0, lpips=0.0)
    )
    assert set(terms) == {"l1"}
    assert total.item() == pytest.approx(l1_loss(pred, target).item(), rel=1e-6)


def test_photometric_loss_combines_terms_with_weights():
    pred, target = _image_pair()
    weights = LossWeights(l1=0.8, l2=0.5, ssim=0.2, lpips=0.0)
    total, terms = photometric_loss(pred, target, weights=weights, ssim_backend="torch")

    assert set(terms) == {"l1", "l2", "ssim"}
    expected = (
        weights.l1 * l1_loss(pred, target)
        + weights.l2 * l2_loss(pred, target)
        + weights.ssim * ssim_loss(pred, target, backend="torch")
    )
    assert total.item() == pytest.approx(expected.item(), rel=1e-6)
    # Reported terms are unweighted, so they stay comparable across runs.
    assert terms["l1"].item() == pytest.approx(l1_loss(pred, target).item(), rel=1e-6)


def test_photometric_loss_defaults_are_the_3dgs_blend():
    pred, target = _image_pair()
    total, terms = photometric_loss(pred, target, ssim_backend="torch")
    assert set(terms) == {"l1", "ssim"}
    expected = 0.8 * l1_loss(pred, target) + 0.2 * ssim_loss(pred, target, backend="torch")
    assert total.item() == pytest.approx(expected.item(), rel=1e-6)


def test_photometric_loss_forwards_the_mask():
    pred, target = _image_pair()
    mask = torch.zeros(_B, _V, _H, _W, dtype=torch.bool)
    mask[..., :8, :] = True
    _, terms = photometric_loss(
        pred, target, weights=LossWeights(l1=1.0, ssim=0.0), mask=mask
    )
    assert terms["l1"].item() == pytest.approx(
        l1_loss(pred[..., :8, :], target[..., :8, :]).item(), rel=1e-6
    )


def test_photometric_loss_rejects_all_zero_weights():
    pred, target = _image_pair()
    with pytest.raises(ValueError, match="nothing to optimize"):
        photometric_loss(pred, target, weights=LossWeights(l1=0.0, l2=0.0, ssim=0.0, lpips=0.0))
