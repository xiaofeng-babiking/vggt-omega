import numpy as np
import pytest

from vggt_omega.evaluates import scene


def _c2w_traj(n):
    # n camera-to-world (4,4) poses, translating along x; identity rotation.
    # A small quadratic y-offset keeps the covariance rank >= 2 so Umeyama
    # alignment (used by CameraPoseMetric with align_scale=True) is well-posed;
    # a purely linear y term would leave the points collinear (rank 1).
    out = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
    xs = np.arange(n, dtype=np.float64)
    out[:, 0, 3] = xs
    out[:, 1, 3] = xs ** 2 * 0.01
    return out


# ---- camera pose guards ------------------------------------------------- #
def test_camera_pose_skipped_without_extrinsics():
    t = _c2w_traj(4)
    assert scene.score_camera_pose(t, t, modalities={"depths"}, num_frames=4) is None


def test_camera_pose_skipped_below_two_frames():
    t = _c2w_traj(1)
    assert scene.score_camera_pose(t, t, modalities={"extrinsics"}, num_frames=1) is None


def test_camera_pose_runs_with_extrinsics():
    gt = _c2w_traj(4)
    pred = _c2w_traj(4)
    res = scene.score_camera_pose(gt, pred, modalities={"extrinsics"}, num_frames=4)
    assert res is not None and "ate" in res and "rpe_trans" in res and "rpe_rot" in res
    assert res["ate"]["rmse"] < 1e-6  # identical trajectories -> ~0 error


# ---- depth scorer: guards + skip --------------------------------------- #
def test_depth_skipped_without_depths_modality():
    gt = np.ones((2, 4, 4), np.float32)
    assert scene.score_depth_frames(gt, gt, modalities={"extrinsics"}) == []


def test_depth_skips_frames_with_no_valid_gt():
    gt = np.ones((3, 4, 4), np.float32)
    gt[1] = 0.0  # frame 1 has no valid GT pixel -> skipped
    pred = np.ones((3, 4, 4), np.float32)
    per_frame = scene.score_depth_frames(gt, pred, modalities={"depths"})
    assert len(per_frame) == 2  # frames 0 and 2 only


def test_depth_frame_has_five_keys():
    gt = np.full((1, 4, 4), 2.0, np.float32)
    pred = np.full((1, 4, 4), 1.0, np.float32)
    (d,) = scene.score_depth_frames(gt, pred, modalities={"depths"})
    assert set(d) == set(scene.DEPTH_KEYS)


# ---- aggregation -------------------------------------------------------- #
def test_aggregate_is_frame_weighted_mean():
    per_frame = [
        {k: float(i + 1) for k in scene.DEPTH_KEYS} for i in range(3)
    ]  # values 1,2,3 per key
    sums, count = scene.depth_sums(per_frame)
    agg = scene.aggregate_depth_from_sums(sums, count)
    assert count == 3
    for k in scene.DEPTH_KEYS:
        assert agg[k] == pytest.approx(2.0)  # mean(1,2,3)
    assert agg["num_frames"] == 3


def test_aggregate_empty_is_none():
    sums, count = scene.depth_sums([])
    assert scene.aggregate_depth_from_sums(sums, count) is None


def test_assemble_metrics_carries_extra():
    m = scene.assemble_metrics("seq", 5, {"ate": 1}, {"abs_rel_mean": 0.1}, resolution=[3, 4])
    assert m == {"scene": "seq", "num_frames": 5, "camera_pose": {"ate": 1},
                 "mono_depth": {"abs_rel_mean": 0.1}, "resolution": [3, 4]}
