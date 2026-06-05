"""Tests for :class:`CameraPoseMetric` on hand-crafted pose trajectories.

Every metric is checked against an *analytically-known* value rather than just
smoke-tested. The constructions exploit three facts:

* **ATE** -- a pure positional scaling by ``s`` about the origin cannot be undone
  by a rigid (``se3``) alignment; the optimal rigid fit is ``R = I``,
  ``t = (1 - s)·centroid``, so the per-pose error is exactly
  ``|1 - s|·‖Cᵢ - centroid‖`` (the scaled cross-covariance is symmetric PSD, so
  Umeyama returns ``R = I``).
* **RPE translation** -- with identity rotations the per-pair relative-translation
  error is just the difference of consecutive translation steps, and it is
  invariant to the global (rigid) alignment.
* **RPE rotation** -- with identity ground-truth rotations and predicted
  ``Rz(θᵢ)``, the per-pair relative-rotation error is the consecutive increment
  ``θ_{i+1} - θᵢ`` in degrees, also alignment-invariant.

Inputs are **camera-to-world** poses (translation column = camera center),
matching :class:`CameraPoseMetric`'s contract (no internal inversion).
"""

from __future__ import annotations

import numpy as np
import pytest

from vggt_omega.evaluates.camera_pose import CameraPoseMetric

# --------------------------------------------------------------------------- #
# hand-crafted dataset builders
# --------------------------------------------------------------------------- #
_STAT_KEYS = ("rmse", "mean", "median", "std", "min", "max")


def _rot_z(deg: float) -> np.ndarray:
    """Rotation matrix about +z by ``deg`` degrees."""
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _poses(rots, centers) -> np.ndarray:
    """Stack rotations ``(N,3,3)`` and centers ``(N,3)`` into c2w ``(N,4,4)``."""
    rots = np.asarray(rots, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    n = rots.shape[0]
    m = np.tile(np.eye(4), (n, 1, 1))
    m[:, :3, :3] = rots
    m[:, :3, 3] = centers
    return m


def _helix_centers(n: int) -> np.ndarray:
    """Non-collinear camera centres (a helix) so Umeyama alignment is well-posed.

    The alignment in ``preprocess`` runs for *all* metrics, so even the RPE
    datasets need a full-rank point cloud -- collinear centres make Umeyama's
    covariance rank-deficient.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), 0.2 * t])


def _make_trajectory(n: int = 10) -> np.ndarray:
    """A smooth, non-degenerate camera-to-world trajectory (rising helix)."""
    rots = [_rot_z(d) for d in np.linspace(0.0, 90.0, n)]
    return _poses(rots, _helix_centers(n))


def _apply_world_similarity(poses, rot=None, trans=None, scale=1.0) -> np.ndarray:
    """Left-apply a global world similarity to camera-to-world poses.

    Under a world map ``x -> scale·rot·x + trans`` the camera-to-world pose
    becomes ``R' = rot·R`` and centre ``C' = scale·rot·C + trans``. Rotations
    stay orthonormal (scale touches only the translation), so the trajectory is
    a similarity of the original camera centres.
    """
    poses = np.asarray(poses, dtype=np.float64)
    rot = np.eye(3) if rot is None else np.asarray(rot, dtype=np.float64)
    trans = np.zeros(3) if trans is None else np.asarray(trans, dtype=np.float64)
    out = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    out[:, :3, :3] = rot @ poses[:, :3, :3]
    out[:, :3, 3] = scale * (poses[:, :3, 3] @ rot.T) + trans
    return out


def _expected_stats(errors) -> dict[str, float]:
    """The six evo statistics for a 1-D array of per-sample errors (std ddof=0)."""
    e = np.asarray(errors, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mean": float(np.mean(e)),
        "median": float(np.median(e)),
        "std": float(np.std(e)),
        "min": float(np.min(e)),
        "max": float(np.max(e)),
    }


def _assert_block(block: dict, errors, tol: float = 1e-9) -> None:
    """Assert a metric block matches the statistics of ``errors``."""
    for key, val in _expected_stats(errors).items():
        assert block[key] == pytest.approx(val, abs=tol), key


# --------------------------------------------------------------------------- #
# wiring / structure
# --------------------------------------------------------------------------- #
def test_run_reports_exactly_the_three_metrics():
    gt = _make_trajectory()
    res = CameraPoseMetric(gt, gt).run()
    assert set(res) == {"ate", "rpe_rot", "rpe_trans"}
    assert len(CameraPoseMetric(gt, gt)) == 3


def test_reduce_ops_selects_reported_statistics():
    gt = _make_trajectory()
    res = CameraPoseMetric(gt, gt, reduce_ops=["rmse", "mean", "sse"]).run()
    for block in res.values():
        assert list(block) == ["rmse", "mean", "sse"]


def test_three_by_four_poses_promoted_to_four_by_four():
    gt = _make_trajectory()
    res = CameraPoseMetric(gt[:, :3, :], gt[:, :3, :]).run()  # (N, 3, 4) input
    assert res["ate"]["rmse"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# ATE  (Absolute Trajectory Error, translation)
# --------------------------------------------------------------------------- #
def test_ate_is_zero_for_identical_trajectories():
    gt = _make_trajectory()
    res = CameraPoseMetric(gt, gt).run()
    _assert_block(res["ate"], np.zeros(len(gt)))


def test_ate_is_zero_after_rigid_alignment_se3():
    # Prediction is a *rigid* world transform of GT -> se3 alignment recovers it.
    gt = _make_trajectory()
    pred = _apply_world_similarity(gt, rot=_rot_z(37.0), trans=[1.0, -2.0, 0.5])
    m = CameraPoseMetric(gt, pred, align_scale=False)
    res = m.run()
    assert res["ate"]["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert m.umeyama_scale == 1.0


def test_ate_zero_and_scale_recovered_after_sim3():
    # Prediction is a *similarity* (scale 3) of GT -> sim3 recovers scale 1/3.
    gt = _make_trajectory()
    s = 3.0
    pred = _apply_world_similarity(gt, rot=_rot_z(20.0), trans=[0.4, 0.1, -0.2], scale=s)
    m = CameraPoseMetric(gt, pred, align_scale=True)
    res = m.run()
    assert res["ate"]["rmse"] == pytest.approx(0.0, abs=1e-6)
    assert m.umeyama_scale == pytest.approx(1.0 / s, rel=1e-6)


def test_ate_scale_drift_under_se3_matches_radial_formula():
    # Pure positional scale by s; rigid alignment cannot absorb it, leaving
    # ATE_i = |1 - s| * ||C_i - centroid||  (R=I, t=(1-s)*centroid optimal).
    rng = np.random.default_rng(0)
    centers = rng.uniform(-2.0, 2.0, size=(8, 3))
    rots = [np.eye(3)] * 8  # rotation is irrelevant to ATE
    gt = _poses(rots, centers)
    s = 1.7
    pred = _poses(rots, s * centers)

    m = CameraPoseMetric(gt, pred, align_scale=False)
    res = m.run()

    radial = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
    _assert_block(res["ate"], abs(1.0 - s) * radial, tol=1e-9)
    assert m.umeyama_scale == 1.0
    # sim3 instead would drive the very same drift to zero.
    res_sim3 = CameraPoseMetric(gt, pred, align_scale=True).run()
    assert res_sim3["ate"]["rmse"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# RPE translation  (Relative Pose Error, frame-to-frame translation)
# --------------------------------------------------------------------------- #
def test_rpe_trans_is_zero_for_identical_trajectories():
    gt = _make_trajectory()
    res = CameraPoseMetric(gt, gt).run()
    _assert_block(res["rpe_trans"], np.zeros(len(gt) - 1))


def test_rpe_trans_matches_known_relative_translation_errors():
    # Identity rotations: the per-pair relative-translation error equals the
    # change in per-pose displacement d_i = pred_i - gt_i, i.e. |d_{i+1} - d_i|.
    # A y-displacement growing by [1,2,3,4] yields exactly those error norms.
    errs = np.array([1.0, 2.0, 3.0, 4.0])
    n = errs.size + 1
    gt_centers = _helix_centers(n)
    disp_y = np.concatenate([[0.0], np.cumsum(errs)])  # [0,1,3,6,10]
    pred_centers = gt_centers + np.column_stack([np.zeros(n), disp_y, np.zeros(n)])
    eye = [np.eye(3)] * n
    gt = _poses(eye, gt_centers)
    pred = _poses(eye, pred_centers)

    res = CameraPoseMetric(gt, pred, rpe_delta=1, align_scale=False).run()
    _assert_block(res["rpe_trans"], errs, tol=1e-9)


def test_rpe_trans_is_invariant_to_global_rigid_transform():
    # RPE compares *relative* poses, so a global rigid transform of the
    # prediction must leave it unchanged.
    errs = np.array([1.0, 2.0, 3.0, 4.0])
    n = errs.size + 1
    gt_centers = _helix_centers(n)
    disp_y = np.concatenate([[0.0], np.cumsum(errs)])
    pred_centers = gt_centers + np.column_stack([np.zeros(n), disp_y, np.zeros(n)])
    eye = [np.eye(3)] * n
    gt = _poses(eye, gt_centers)
    pred = _poses(eye, pred_centers)
    pred_moved = _apply_world_similarity(pred, rot=_rot_z(55.0), trans=[3.0, 1.0, -1.0])

    base = CameraPoseMetric(gt, pred, align_scale=False).run()["rpe_trans"]
    moved = CameraPoseMetric(gt, pred_moved, align_scale=False).run()["rpe_trans"]
    for key in _STAT_KEYS:
        assert moved[key] == pytest.approx(base[key], abs=1e-9), key


# --------------------------------------------------------------------------- #
# RPE rotation  (Relative Pose Error, frame-to-frame rotation, degrees)
# --------------------------------------------------------------------------- #
def test_rpe_rot_is_zero_for_identical_trajectories():
    gt = _make_trajectory()
    res = CameraPoseMetric(gt, gt).run()
    _assert_block(res["rpe_rot"], np.zeros(len(gt) - 1))


def test_rpe_rot_matches_known_relative_rotation_errors():
    # GT rotations identity; predicted yaw Rz(theta_i) with increments [1,2,3,4]
    # degrees -> per-pair relative-rotation error == those increments.
    incr = np.array([1.0, 2.0, 3.0, 4.0])
    theta = np.concatenate([[0.0], np.cumsum(incr)])  # [0,1,3,6,10] deg, 5 poses
    n = theta.size
    centers = _helix_centers(n)  # non-collinear, identical for gt/pred
    gt = _poses([np.eye(3)] * n, centers)
    pred = _poses([_rot_z(t) for t in theta], centers)

    res = CameraPoseMetric(gt, pred, rpe_delta=1, align_scale=False).run()
    _assert_block(res["rpe_rot"], incr, tol=1e-7)


def test_rpe_rot_respects_frame_gap_delta():
    # evo's RPE (all_pairs=False) uses *non-overlapping* pairs stepping by delta:
    # delta=2 -> pairs (0,2),(2,4) over theta=[0,1,3,6,10] -> errors [3,7] deg.
    incr = np.array([1.0, 2.0, 3.0, 4.0])
    theta = np.concatenate([[0.0], np.cumsum(incr)])
    n = theta.size
    centers = _helix_centers(n)
    gt = _poses([np.eye(3)] * n, centers)
    pred = _poses([_rot_z(t) for t in theta], centers)

    res = CameraPoseMetric(gt, pred, rpe_delta=2, align_scale=False).run()
    expected = np.diff(theta[::2])  # pairs (0,2),(2,4) -> [3, 7]
    _assert_block(res["rpe_rot"], expected, tol=1e-7)


# --------------------------------------------------------------------------- #
# visualization
# --------------------------------------------------------------------------- #
def test_visualize_writes_trajectory_and_ate_plots(tmp_path):
    gt = _make_trajectory()
    pred = _apply_world_similarity(gt, scale=1.3)  # varied per-pose ATE
    prefix = str(tmp_path / "seq01")
    CameraPoseMetric(gt, pred, align_scale=False).run(vis_path=prefix)
    for suffix in ("_traj.png", "_ate.png"):
        f = tmp_path / f"seq01{suffix}"
        assert f.exists() and f.stat().st_size > 0


def test_visualize_handles_degenerate_constant_error(tmp_path):
    # Identical trajectories -> per-pose ATE is constant zero; the colormap span
    # guard must keep the plot from blowing up.
    gt = _make_trajectory()
    prefix = str(tmp_path / "flat")
    CameraPoseMetric(gt, gt).run(vis_path=prefix)
    assert (tmp_path / "flat_ate.png").exists()


def test_run_without_vis_path_writes_nothing(tmp_path):
    gt = _make_trajectory()
    CameraPoseMetric(gt, gt).run()
    assert not list(tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# input validation (check())
# --------------------------------------------------------------------------- #
def test_too_few_poses_raise():
    gt = _make_trajectory(2)
    with pytest.raises(AssertionError):
        CameraPoseMetric(gt, gt).run()


def test_shape_mismatch_raises():
    gt = _make_trajectory(10)
    with pytest.raises(AssertionError):
        CameraPoseMetric(gt, gt[:6]).run()


def test_bad_pose_shape_raises():
    bad = np.zeros((10, 2, 4))
    with pytest.raises(AssertionError):
        CameraPoseMetric(bad, bad).run()


def test_non_finite_raises():
    gt = _make_trajectory(10)
    bad = gt.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(AssertionError):
        CameraPoseMetric(bad, gt).run()


def test_unknown_reduce_op_raises():
    gt = _make_trajectory(10)
    with pytest.raises(AssertionError):
        CameraPoseMetric(gt, gt, reduce_ops=["bogus"]).run()
