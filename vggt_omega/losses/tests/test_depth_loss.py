"""Tests for the depth loss against analytically-known values.

With GT depth 1 everywhere the relative-scale factor is exactly
``1 + 1/1 = 2``, so a constant prediction offset ``δ`` with confidence ``c``
gives a main term of ``2cδ``; a constant residual has zero spatial gradient;
confidence 1 makes the ``−α log c`` term vanish.
"""

from __future__ import annotations

import math

import pytest
import torch

from vggt_omega.losses import depth_loss


def _const(value: float, shape=(1, 1, 2, 2)) -> torch.Tensor:
    return torch.full(shape, value)


def test_perfect_prediction_with_unit_confidence_is_zero():
    gt = _const(1.0)
    loss = depth_loss(gt[..., None], _const(1.0), gt, torch.ones_like(gt, dtype=torch.bool))
    assert loss.item() == pytest.approx(0.0)


def test_constant_offset_value():
    gt = _const(1.0)
    pred = _const(1.25)
    conf = _const(3.0)
    valid = torch.ones_like(gt, dtype=torch.bool)
    # main = c * (1 + 1/D) * |e| = 3 * 2 * 0.25; gradient term 0; -alpha*log(3).
    expected = 3.0 * 2.0 * 0.25 - 0.2 * math.log(3.0)
    assert depth_loss(pred, conf, gt, valid, alpha=0.2).item() == pytest.approx(expected, rel=1e-6)


def test_alpha_zero_disables_uncertainty_term():
    gt = _const(1.0)
    conf = _const(math.e)
    valid = torch.ones_like(gt, dtype=torch.bool)
    assert depth_loss(gt, conf, gt, valid, alpha=0.0).item() == pytest.approx(0.0)
    assert depth_loss(gt, conf, gt, valid, alpha=0.2).item() == pytest.approx(-0.2)


def test_gradient_term_on_linear_ramp():
    # 1x2 image, GT 1 everywhere, residual e = [0, g]: main term mean is
    # 2*g/2 = g, the single x-gradient is g, no y-gradients. Total 2g.
    g = 0.5
    gt = _const(1.0, (1, 1, 1, 2))
    pred = gt.clone()
    pred[..., 1] += g
    valid = torch.ones_like(gt, dtype=torch.bool)
    loss = depth_loss(pred, _const(1.0, gt.shape), gt, valid, alpha=0.0)
    assert loss.item() == pytest.approx(2 * g, rel=1e-6)


def test_invalid_pixels_contribute_nothing():
    gt = torch.tensor([[1.0, 0.0]]).view(1, 1, 1, 2)
    pred = torch.tensor([[1.0, 99.0]]).view(1, 1, 1, 2)
    valid = gt > 0
    # The huge error on the invalid pixel is masked; its x-gradient pair is
    # also invalid because one endpoint is invalid.
    assert depth_loss(pred, _const(1.0, gt.shape), gt, valid, alpha=0.0).item() == pytest.approx(0.0)


def test_accepts_trailing_channel_prediction():
    gt = _const(1.0)
    valid = torch.ones_like(gt, dtype=torch.bool)
    with_channel = depth_loss(gt[..., None] * 1.5, _const(1.0), gt, valid, alpha=0.0)
    without_channel = depth_loss(gt * 1.5, _const(1.0), gt, valid, alpha=0.0)
    assert with_channel.item() == pytest.approx(without_channel.item())


def test_near_depth_weighting_exceeds_far():
    # Identical relative error: D=0.5 has weight 3, D=2 has weight 1.5.
    valid = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    near = depth_loss(_const(0.6, (1, 1, 1, 1)), _const(1.0, (1, 1, 1, 1)), _const(0.5, (1, 1, 1, 1)), valid, alpha=0.0)
    far = depth_loss(_const(2.4, (1, 1, 1, 1)), _const(1.0, (1, 1, 1, 1)), _const(2.0, (1, 1, 1, 1)), valid, alpha=0.0)
    assert near.item() == pytest.approx(3.0 * 0.1, rel=1e-5)
    assert far.item() == pytest.approx(1.5 * 0.4, rel=1e-5)
