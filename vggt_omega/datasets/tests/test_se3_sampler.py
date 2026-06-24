"""Tests for SE(3) arc-length frame sampling (sample_se3_trajectory).

The sampler reads poses via ``seq.get_poses(sensor)`` (a NumpySE3Trajectory) and
places ``num`` frames at equal SE(3) arc-length over a window. A tiny in-memory
``_PoseSeq`` stand-in supplies ``get_length`` / ``get_poses`` so the laws can run
on synthetic trajectories; a final class runs the sampler end-to-end on a real
:class:`TumSequence` to confirm compatibility with the live sequence API.
"""
import os
import tempfile

import numpy as np
import pytest

from vggt_omega.datasets.samplers import sample_se3_trajectory
from vggt_omega.datasets.utils.se3_trajectory import NumpySE3Trajectory

N_RANDOM = 30
randomized = pytest.mark.parametrize("seed", range(N_RANDOM))

_SENSOR = 0


class _PoseSeq:
    """Minimal BaseSequence-like pose source: exposes the two methods the sampler
    uses -- ``get_length(sensor)`` and ``get_poses(sensor)`` -- over an in-memory
    :class:`NumpySE3Trajectory`."""

    def __init__(self, traj: NumpySE3Trajectory):
        self._traj = traj

    def get_length(self, sensor_id):
        return len(self._traj)

    def get_poses(self, sensor_id):
        return self._traj


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_quat(rng) -> np.ndarray:
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def random_trans(rng, scale=5.0) -> np.ndarray:
    return rng.uniform(-scale, scale, size=3)


def _traj(rows) -> NumpySE3Trajectory:
    """Build a trajectory from ``[qx, qy, qz, qw, tx, ty, tz]`` rows."""
    rows = np.asarray(rows, dtype=float).reshape(-1, 7)
    return NumpySE3Trajectory(
        timestamps=np.arange(len(rows), dtype=float), poses=rows, source=0, target="w"
    )


def _random_rows(rng, n):
    """``n`` random pose rows; quat then trans per frame (stable draw order)."""
    return [np.concatenate([random_quat(rng), random_trans(rng)]) for _ in range(n)]


def _still_then_move_rows(rng, n_still, n_move):
    q0, t0 = random_quat(rng), random_trans(rng)
    still = np.concatenate([q0, t0])
    return [still.copy() for _ in range(n_still)] + _random_rows(rng, n_move)


def _gaps(traj) -> np.ndarray:
    """Per-frame SE(3) twist-norm gaps along a trajectory, ``(N-1,)``."""
    return np.linalg.norm(traj.consecutive_twist(), axis=1)


def _sample(rows, num, **kw):
    return sample_se3_trajectory(_PoseSeq(_traj(rows)), _SENSOR, num, **kw)


def _sample_traj(traj, num, **kw):
    return sample_se3_trajectory(_PoseSeq(traj), _SENSOR, num, **kw)


# --------------------------------------------------------------------------- #
# shape / endpoints / monotonicity / return contract
# --------------------------------------------------------------------------- #
@randomized
def test_return_contract(seed):
    rng = _rng(seed)
    traj = _traj(_random_rows(rng, 12))
    idx, dist = _sample_traj(traj, num=5, start=0, end=11)
    # equal-length lists; distances[0] == 0.0
    assert len(idx) == len(dist)
    assert dist[0] == 0.0
    assert all(isinstance(i, int) for i in idx)
    assert all(isinstance(d, float) for d in dist)
    # endpoints pinned; strictly increasing (duplicates removed)
    assert idx[0] == 0 and idx[-1] == 11
    assert all(idx[k] < idx[k + 1] for k in range(len(idx) - 1))
    assert all(0 <= i < len(traj) for i in idx)
    # distances are the path arc-length (sum of per-frame gaps) between kept idx
    cum = np.concatenate([[0.0], np.cumsum(_gaps(traj))])
    for k in range(1, len(idx)):
        np.testing.assert_allclose(dist[k], cum[idx[k]] - cum[idx[k - 1]], atol=1e-9)


@randomized
def test_count_at_most_num(seed):
    # de-duplication can only shrink the result, never exceed num.
    idx, dist = _sample(_random_rows(_rng(seed), 12), num=5, start=0, end=11)
    assert 2 <= len(idx) <= 5
    assert len(dist) == len(idx)


@randomized
def test_respects_subrange(seed):
    idx, _ = _sample(_random_rows(_rng(seed), 15), num=4, start=3, end=10)
    assert idx[0] == 3 and idx[-1] == 10
    assert all(3 <= i <= 10 for i in idx)


def test_default_window_is_full():
    idx, _ = _sample(_random_rows(_rng(0), 8), num=4)  # start=0, end=-1
    assert idx[0] == 0 and idx[-1] == 7


def test_negative_end_index():
    rows = _random_rows(_rng(0), 10)
    assert _sample(rows, num=3, start=0, end=-1)[0][-1] == 9
    assert _sample(rows, num=3, start=0, end=-3)[0][-1] == 7


def test_num_one_returns_start():
    idx, dist = _sample(_random_rows(_rng(0), 6), num=1, start=2, end=5)
    assert idx == [2] and dist == [0.0]


def test_num_equals_window_returns_every_frame():
    idx, dist = _sample(_random_rows(_rng(0), 6), num=4, start=1, end=4)
    assert idx == [1, 2, 3, 4]
    assert len(dist) == 4 and dist[0] == 0.0


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_asserts_window_at_least_num():
    with pytest.raises(AssertionError):
        _sample(_random_rows(_rng(0), 10), num=5, start=0, end=3)  # window 4 < 5


def test_rejects_nonpositive_num():
    with pytest.raises(ValueError):
        _sample(_random_rows(_rng(0), 5), num=0)


def test_rejects_end_before_start():
    with pytest.raises(ValueError):
        _sample(_random_rows(_rng(0), 6), num=2, start=4, end=1)


def test_rejects_out_of_range_window():
    with pytest.raises(ValueError):
        _sample(_random_rows(_rng(0), 5), num=2, start=0, end=99)


def test_rejects_empty():
    with pytest.raises(ValueError):
        _sample([], num=1)


# --------------------------------------------------------------------------- #
# arc-length adaptivity (the core behaviour)
# --------------------------------------------------------------------------- #
@randomized
def test_skips_small_motion_frames(seed):
    # 4 near-static frames then 5 moving frames: at most one interior sample should
    # fall in the static region; the rest land on the moving stretch.
    traj = _traj(_still_then_move_rows(_rng(seed), 4, 5))
    idx, _ = _sample_traj(traj, num=5, start=0, end=len(traj) - 1)
    interior_in_static = sum(1 for i in idx[1:-1] if i < 4)
    assert interior_in_static <= 1


@randomized
def test_equal_arclength_spacing(seed):
    # Selected frames are ~equally spaced in cumulative SE(3) arc-length.
    idx, dist = _sample(_random_rows(_rng(seed), 16), num=5, start=0, end=15)
    seg = dist[1:]  # the returned distances ARE the inter-sample arc-lengths
    ideal = sum(seg) / len(seg)
    assert all(s > 0 for s in seg)
    # snapping to discrete frames is approximate -> loose tolerance.
    assert max(abs(s - ideal) for s in seg) < 0.6 * ideal


@randomized
def test_dedup_removes_duplicate_snaps(seed):
    # A long static head packed with many targets must collapse to unique indices.
    traj = _traj(_still_then_move_rows(_rng(seed), 8, 2))
    idx, dist = _sample_traj(traj, num=8, start=0, end=len(traj) - 1)
    assert len(idx) == len(set(idx))  # no duplicates
    assert len(idx) == len(dist)
    assert idx[0] == 0 and idx[-1] == len(traj) - 1


def test_no_motion_spreads_by_index():
    # All-equal poses -> zero arc-length -> even index spacing, all gaps zero.
    row = np.concatenate([random_quat(_rng(0)), random_trans(_rng(0))])
    idx, dist = _sample([row.copy() for _ in range(9)], num=5, start=0, end=8)
    assert idx == [0, 2, 4, 6, 8]
    assert dist == [0.0, 0.0, 0.0, 0.0, 0.0]


@randomized
def test_deterministic_for_fixed_window(seed):
    rows = _random_rows(_rng(seed), 12)
    assert _sample(rows, num=5, start=1, end=10) == _sample(rows, num=5, start=1, end=10)


def test_ignores_out_of_window_poses():
    # The sampler must depend ONLY on poses in [start, end]: perturbing frames
    # outside the window leaves the result unchanged.
    rng = _rng(0)
    rows = _random_rows(rng, 50)
    base = _sample(rows, num=4, start=10, end=19)
    perturbed = [r.copy() for r in rows]
    for i in list(range(0, 10)) + list(range(20, 50)):
        perturbed[i] = np.concatenate([random_quat(rng), random_trans(rng)])
    assert _sample(perturbed, num=4, start=10, end=19) == base


# --------------------------------------------------------------------------- #
# end-to-end against the live TUM sequence API
# --------------------------------------------------------------------------- #
def _make_tum(n=8):
    # Reuse the synthetic TUM builder from the sequence tests.
    from vggt_omega.datasets.tests.test_sequences import _write_tum_sequence
    from vggt_omega.datasets.sequences.tum import TumSequence

    root = tempfile.mkdtemp()
    seq_id = "rgbd_dataset_freiburg1_xyz"
    _write_tum_sequence(root, seq_id, n=n)
    return TumSequence(root, seq_id)


def test_sampler_on_tum_sequence():
    s = _make_tum(n=8)
    sid = s.get_sensors()[0]
    idx, dist = sample_se3_trajectory(s, sid, num=4, start=0, end=7)
    # endpoints pinned, strictly increasing, in range
    assert idx[0] == 0 and idx[-1] == 7
    assert all(idx[k] < idx[k + 1] for k in range(len(idx) - 1))
    assert 2 <= len(idx) <= 4 and len(dist) == len(idx)
    # TUM camera moves at unit speed per frame (translation [k,0,0]); each unit
    # frame-step contributes a twist norm of 1, so the full window arc-length is 7.
    assert np.isclose(sum(dist), 7.0, atol=1e-3)


def test_sampler_matches_manual_arclength_on_tum():
    s = _make_tum(n=8)
    sid = s.get_sensors()[0]
    traj = s.get_poses(sid)
    cum = np.concatenate([[0.0], np.cumsum(_gaps(traj))])
    idx, dist = sample_se3_trajectory(s, sid, num=4, start=0, end=7)
    for k in range(1, len(idx)):
        np.testing.assert_allclose(dist[k], cum[idx[k]] - cum[idx[k - 1]], atol=1e-9)
