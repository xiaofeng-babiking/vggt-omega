"""Tests for :class:`MonoDepthMetric` on hand-crafted depth maps.

Each metric is checked against an analytically-known value:

* a constant multiplicative factor ``f`` between prediction and GT makes the
  per-pixel relative error ``|pred - gt| / gt == |f - 1|`` exactly (the GT
  cancels), so ``abs_rel["mean"] == |f - 1|`` and every other reduction equals it
  too; the worst-direction ratio is ``max(f, 1/f)``, fixing the ``delta``
  accuracies;
* median alignment recovers a pure global scale exactly, driving ``abs_rel`` to 0.

Depth maps use the dataset convention: ``0`` = invalid, ``< 0`` = sky; only
strictly-positive, finite pixels are scored. Argument order is **GT first**.
"""

from __future__ import annotations

import numpy as np
import pytest

from vggt_omega.evaluates.mono_depth import MonoDepthMetric


def _depth(rows) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _smooth_depth(h: int = 16, w: int = 16) -> np.ndarray:
    """A smooth, strictly-positive depth map for visualization tests."""
    yy, xx = np.mgrid[0:h, 0:w]
    return 1.0 + 0.1 * xx + 0.05 * yy


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def test_reports_exactly_two_metric_families():
    gt = _depth([[1, 2], [3, 4]])
    res = MonoDepthMetric(gt, gt, align="none").run()
    assert set(res) == {"abs_rel", "delta"}
    assert len(MonoDepthMetric(gt, gt)) == 2


def test_reduce_ops_selects_abs_rel_statistics():
    gt = _depth([[1, 2], [3, 4]])
    res = MonoDepthMetric(gt, 1.2 * gt, align="none", reduce_ops=["mean", "max"]).run()
    assert list(res["abs_rel"]) == ["mean", "max"]
    assert set(res["delta"]) == {"delta1", "delta2", "delta3"}


# --------------------------------------------------------------------------- #
# abs_rel
# --------------------------------------------------------------------------- #
def test_identical_prediction_is_perfect():
    gt = _depth([[1, 2], [3, 4]])
    res = MonoDepthMetric(gt, gt, align="none").run()
    assert res["abs_rel"]["mean"] == pytest.approx(0.0)
    assert res["abs_rel"]["max"] == pytest.approx(0.0)
    assert res["delta"]["delta1"] == pytest.approx(1.0)


def test_abs_rel_equals_constant_scale_offset():
    # pred = f*gt -> |pred-gt|/gt = |f-1| at every pixel, so all stats == |f-1|.
    gt = _depth([[2, 4], [6, 8]])
    res = MonoDepthMetric(gt, 1.1 * gt, align="none").run()
    assert res["abs_rel"]["mean"] == pytest.approx(0.1)  # |1.1 - 1|
    assert res["abs_rel"]["max"] == pytest.approx(0.1)
    assert res["abs_rel"]["std"] == pytest.approx(0.0)
    assert res["delta"]["delta1"] == pytest.approx(1.0)  # ratio 1.1 < 1.25


def test_abs_rel_and_delta_known_for_mixed_factors():
    gt = _depth([[5, 5], [5, 5]])
    factors = _depth([[1.0, 1.2], [1.3, 2.0]])
    res = MonoDepthMetric(gt, gt * factors, align="none").run()
    # per-pixel |f-1| = [0, 0.2, 0.3, 1.0]
    assert res["abs_rel"]["mean"] == pytest.approx((0.0 + 0.2 + 0.3 + 1.0) / 4)  # 0.375
    assert res["abs_rel"]["max"] == pytest.approx(1.0)
    # ratios [1, 1.2, 1.3, 2.0]: <1.25 -> 2/4; <1.5625 -> 3/4
    assert res["delta"]["delta1"] == pytest.approx(0.5)
    assert res["delta"]["delta2"] == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# scale alignment
# --------------------------------------------------------------------------- #
def test_median_alignment_recovers_global_scale():
    gt = _depth([[2, 4], [6, 9]])
    m = MonoDepthMetric(gt, 0.25 * gt, align="median")
    res = m.run()
    assert m.depth_scale == pytest.approx(4.0)  # median(gt)/median(gt/4)
    assert res["abs_rel"]["mean"] == pytest.approx(0.0)
    assert res["delta"]["delta1"] == pytest.approx(1.0)


def test_align_none_keeps_the_scale_error():
    gt = _depth([[2, 4], [6, 8]])
    m = MonoDepthMetric(gt, 0.5 * gt, align="none")
    res = m.run()
    assert res["abs_rel"]["mean"] == pytest.approx(0.5)  # |0.5 - 1|
    assert res["delta"]["delta1"] == pytest.approx(0.0)  # ratio 2.0 >= 1.25
    assert m.depth_scale == 1.0


# --------------------------------------------------------------------------- #
# validity masking
# --------------------------------------------------------------------------- #
def test_invalid_and_sky_pixels_are_excluded():
    gt = _depth([[4, 0], [-2, 4]])  # 0 = invalid, -2 = sky
    pred = _depth([[4, 9], [9, 4]])  # the masked pixels would corrupt abs_rel
    m = MonoDepthMetric(gt, pred, align="none")
    res = m.run()
    assert m.num_valid == 2
    assert res["abs_rel"]["mean"] == pytest.approx(0.0)  # only the matching 4s scored


def test_mask_argument_restricts_scored_pixels():
    gt = _depth([[5, 5], [5, 5]])
    pred = _depth([[5, 99], [99, 5]])
    mask = np.array([[True, False], [False, True]])
    m = MonoDepthMetric(gt, pred, mask=mask, align="none")
    res = m.run()
    assert m.num_valid == 2
    assert res["abs_rel"]["mean"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# visualization
# --------------------------------------------------------------------------- #
def test_visualize_writes_depth_panel(tmp_path):
    gt = _smooth_depth()
    pred = gt * 1.2
    MonoDepthMetric(gt, pred).run(vis_path=str(tmp_path / "d"))
    f = tmp_path / "d_depth.png"
    assert f.exists() and f.stat().st_size > 0


def test_run_without_vis_path_writes_nothing(tmp_path):
    gt = _smooth_depth()
    MonoDepthMetric(gt, gt).run()
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# validation (check / preprocess)
# --------------------------------------------------------------------------- #
def test_no_valid_pixels_raises():
    gt = np.zeros((3, 3))  # every pixel invalid
    with pytest.raises(AssertionError):
        MonoDepthMetric(gt, np.ones((3, 3))).run()


def test_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        MonoDepthMetric(np.ones((4, 4)), np.ones((4, 5))).run()


def test_non_2d_input_raises():
    with pytest.raises(AssertionError):
        MonoDepthMetric(np.ones(4), np.ones(4)).run()


def test_unknown_align_raises():
    gt = _depth([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        MonoDepthMetric(gt, gt, align="bogus").run()


def test_mask_shape_mismatch_raises():
    gt = _depth([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        MonoDepthMetric(gt, gt, mask=np.ones((3, 3), dtype=bool)).run()


def test_delta_threshold_must_exceed_one():
    gt = _depth([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        MonoDepthMetric(gt, gt, delta_threshold=0.9).run()


def test_unknown_reduce_op_raises():
    gt = _depth([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        MonoDepthMetric(gt, gt, reduce_ops=["bogus"]).run()
