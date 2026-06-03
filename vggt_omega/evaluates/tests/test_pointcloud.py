"""Tests for :class:`PointcloudMetric` on hand-crafted clouds.

The constructions give analytically-known answers:

* identical clouds -> zero accuracy/completeness and full normal consistency;
* a *constant* vertical offset of the same points -> every nearest-neighbor
  distance is exactly the offset (each shifted point's nearest neighbor is its
  own original), so accuracy == completeness == offset for every reduction;
* a known similarity (scale + rotation + translation) of a distinctive shape is
  recovered by ICP, driving accuracy to ~0 and the scale to its inverse.

Each metric is a dict of ``reduce_ops`` statistics (``mean`` is the headline;
``median`` is the DTU reduction); ``fscore`` is a ``{precision, recall, fscore}``
block. Argument order is **GT first**.
"""

from __future__ import annotations

import numpy as np
import pytest

from vggt_omega.evaluates.pointcloud import PointcloudMetric


def _plane(n: int, seed: int = 0) -> np.ndarray:
    """A flat patch of points on the z=0 plane in the unit square."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    return np.column_stack([xy, np.zeros(n)])


def _helix(n: int) -> np.ndarray:
    """A helix: a distinctive, asymmetric shape with unambiguous ICP matches."""
    t = np.linspace(0.0, 6.0 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), 0.3 * t])


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def test_reports_expected_metric_keys():
    pts = _plane(200)
    res = PointcloudMetric(pts, pts).run()
    assert set(res) == {
        "accuracy",
        "completeness",
        "chamfer",
        "normal_consistency",
        "fscore",
    }
    assert len(PointcloudMetric(pts, pts)) == 5
    assert set(res["fscore"]) == {"precision", "recall", "fscore"}


def test_reduce_ops_selects_distance_statistics():
    pts = _plane(200)
    res = PointcloudMetric(pts, pts, reduce_ops=["mean", "median"]).run()
    assert list(res["accuracy"]) == ["mean", "median"]
    assert list(res["chamfer"]) == ["mean", "median"]


# --------------------------------------------------------------------------- #
# distances
# --------------------------------------------------------------------------- #
def test_identical_clouds_zero_distance_and_full_normal_consistency():
    pts = _plane(500)
    m = PointcloudMetric(pts, pts)
    res = m.run()
    assert res["accuracy"]["mean"] == pytest.approx(0.0)
    assert res["completeness"]["mean"] == pytest.approx(0.0)
    assert res["chamfer"]["mean"] == pytest.approx(0.0)
    # A plane's PCA normals all point along z (up to sign) -> |cos| == 1.
    assert res["normal_consistency"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert m.num_pred == 500 and m.num_gt == 500
    assert m.icp_scale == pytest.approx(1.0)


def test_constant_offset_sets_accuracy_and_completeness():
    pts = _plane(800)
    shifted = pts + np.array([0.0, 0.0, 0.1])  # lift the whole plane by 0.1
    res = PointcloudMetric(pts, shifted).run()  # gt=pts, pred=shifted
    # Each shifted point's nearest neighbor is its own original -> gap is exact,
    # so every reduction of the distance equals 0.1.
    assert res["accuracy"]["mean"] == pytest.approx(0.1, abs=1e-6)
    assert res["accuracy"]["min"] == pytest.approx(0.1, abs=1e-6)
    assert res["accuracy"]["max"] == pytest.approx(0.1, abs=1e-6)
    assert res["completeness"]["mean"] == pytest.approx(0.1, abs=1e-6)
    assert res["normal_consistency"]["mean"] == pytest.approx(1.0, abs=1e-6)


def test_chamfer_is_mean_of_accuracy_and_completeness():
    rng = np.random.default_rng(1)
    gt = _plane(300)
    pred = gt + rng.normal(0.0, 0.02, size=gt.shape)
    res = PointcloudMetric(gt, pred).run()
    assert res["chamfer"]["mean"] == pytest.approx(
        0.5 * (res["accuracy"]["mean"] + res["completeness"]["mean"])
    )


def test_median_reduction_is_robust_to_an_outlier():
    pts = _plane(500)
    pred = pts.copy()
    pred[0] += np.array([0.0, 0.0, 10.0])  # one gross outlier
    res = PointcloudMetric(pts, pred).run()  # mean and median in one pass
    assert res["accuracy"]["median"] < res["accuracy"]["mean"]
    assert res["accuracy"]["median"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# normal consistency
# --------------------------------------------------------------------------- #
def test_supplied_normals_drive_normal_consistency():
    pts = _plane(300)
    up = np.tile([0.0, 0.0, 1.0], (300, 1))
    flipped = np.tile([1.0, 0.0, 0.0], (300, 1))  # orthogonal to `up`
    same = PointcloudMetric(pts, pts, gt_normals=up, pred_normals=up).run()
    orth = PointcloudMetric(pts, pts, gt_normals=flipped, pred_normals=up).run()
    assert same["normal_consistency"]["mean"] == pytest.approx(1.0)
    assert orth["normal_consistency"]["mean"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# F-score (threshold-gated)
# --------------------------------------------------------------------------- #
def test_fscore_perfect_within_threshold_and_zero_beyond():
    pts = _plane(400)
    near = pts + np.array([0.0, 0.0, 0.01])
    res = PointcloudMetric(pts, near, threshold=0.05).run()["fscore"]
    # Every point is within 0.05 -> perfect precision/recall/F-score.
    assert res["precision"] == pytest.approx(1.0)
    assert res["recall"] == pytest.approx(1.0)
    assert res["fscore"] == pytest.approx(1.0)

    strict = PointcloudMetric(pts, near, threshold=0.005).run()["fscore"]
    # Gap (0.01) exceeds threshold -> nothing counts.
    assert strict["fscore"] == pytest.approx(0.0)


def test_fscore_is_nan_without_threshold():
    pts = _plane(200)
    res = PointcloudMetric(pts, pts).run()["fscore"]
    assert np.isnan(res["precision"])
    assert np.isnan(res["recall"])
    assert np.isnan(res["fscore"])


# --------------------------------------------------------------------------- #
# ICP registration
# --------------------------------------------------------------------------- #
def test_icp_recovers_similarity_transform():
    gt = _helix(3000)
    # A known scale + rotation + translation of the prediction; ICP's moment
    # init plus the helix's distinctive shape recover it.
    R = _rot_z(0.25)
    pred = (1.5 * (R @ gt.T)).T + np.array([0.5, -0.3, 0.2])

    raw = PointcloudMetric(gt, pred, align="none").run()
    m = PointcloudMetric(gt, pred, align="icp", align_scale=True)
    aligned = m.run()

    assert aligned["accuracy"]["mean"] < 1e-2
    assert aligned["accuracy"]["mean"] < raw["accuracy"]["mean"]
    assert m.icp_scale == pytest.approx(1.0 / 1.5, rel=5e-2)


# --------------------------------------------------------------------------- #
# input handling / validation
# --------------------------------------------------------------------------- #
def test_non_finite_rows_are_dropped():
    pts = _plane(100)
    polluted = np.vstack([pts, np.full((5, 3), np.nan)])
    m = PointcloudMetric(pts, polluted)
    m.run()
    assert m.num_pred == 100  # the 5 NaN rows are filtered out


def test_visualize_writes_cloud_plot(tmp_path):
    gt = _plane(300)
    pred = gt + np.array([0.0, 0.0, 0.05])
    PointcloudMetric(gt, pred, threshold=0.1).run(vis_path=str(tmp_path / "pc"))
    f = tmp_path / "pc_cloud.png"
    assert f.exists() and f.stat().st_size > 0


def test_bad_shape_raises():
    with pytest.raises(AssertionError):
        PointcloudMetric(np.zeros((10, 2)), np.zeros((10, 3))).run()


def test_too_few_points_raise():
    with pytest.raises(AssertionError):
        PointcloudMetric(np.zeros((2, 3)), np.zeros((2, 3))).run()


def test_unknown_align_raises():
    pts = _plane(50)
    with pytest.raises(AssertionError):
        PointcloudMetric(pts, pts, align="bogus").run()


def test_nonpositive_threshold_raises():
    pts = _plane(50)
    with pytest.raises(AssertionError):
        PointcloudMetric(pts, pts, threshold=-1.0).run()


def test_unknown_reduce_op_raises():
    pts = _plane(50)
    with pytest.raises(AssertionError):
        PointcloudMetric(pts, pts, reduce_ops=["bogus"]).run()


def test_normals_shape_mismatch_raises():
    pts = _plane(50)
    with pytest.raises(AssertionError):
        PointcloudMetric(pts, pts, gt_normals=np.ones((10, 3))).run()
