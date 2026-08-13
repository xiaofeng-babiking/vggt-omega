"""Tests for SE(3) motion-weighted random frame sampling (sample_se3_random).

The sampler draws exactly ``num`` distinct frames WITHOUT replacement, with
per-frame probability proportional to the arc-length each frame owns on the
window's SE(3) arc-length axis (its Voronoi cell), plus a tiny uniform floor.
Fixtures are shared with the equal-arc sampler tests (test_se3_sampler)."""
import tempfile

import numpy as np
import pytest

from vggt_omega.datasets.samplers import sample_se3_random
from vggt_omega.datasets.tests.test_se3_sampler import (
    _PoseSeq,
    _gaps,
    _random_rows,
    _rng,
    _still_then_move_rows,
    _traj,
)

_SENSOR = 0


def _sample(rows, num, **kw):
    return sample_se3_random(_PoseSeq(_traj(rows)), _SENSOR, num, **kw)


def _const_velocity_rows(n):
    """Identity rotation, translation [k, 0, 0]: every per-frame gap is exactly
    1.0, so arc-length degenerates to the index axis and the ONLY variance in
    inter-sample distances comes from the random draw itself."""
    return [np.array([0.0, 0.0, 0.0, 1.0, float(k), 0.0, 0.0]) for k in range(n)]


# --------------------------------------------------------------------------- #
# contract: exactly-num / distinct / sorted / in-window / distances semantics
# --------------------------------------------------------------------------- #
def test_return_contract():
    np.random.seed(0)
    traj = _traj(_random_rows(_rng(0), 12))
    idx, dist = sample_se3_random(_PoseSeq(traj), _SENSOR, 5, start=0, end=11)
    assert len(idx) == 5 and len(dist) == 5          # exactly num, always
    assert len(set(idx)) == 5                        # distinct
    assert all(isinstance(i, int) for i in idx)
    assert all(isinstance(d, float) for d in dist)
    assert all(idx[k] < idx[k + 1] for k in range(len(idx) - 1))
    assert all(0 <= i <= 11 for i in idx)
    assert dist[0] == 0.0
    # distances are the path arc-length between kept indices
    cum = np.concatenate([[0.0], np.cumsum(_gaps(traj))])
    for k in range(1, len(idx)):
        np.testing.assert_allclose(dist[k], cum[idx[k]] - cum[idx[k - 1]], atol=1e-9)


def test_exactly_num_even_when_equal_arc_would_dedup():
    # A long static head makes the equal-arc sampler shrink its output; the
    # weighted draw must still return exactly num distinct frames.
    np.random.seed(0)
    rows = _still_then_move_rows(_rng(0), 8, 2)
    idx, dist = _sample(rows, num=8, start=0, end=9)
    assert len(idx) == 8 and len(set(idx)) == 8 and len(dist) == 8


def test_respects_subrange():
    np.random.seed(0)
    idx, _ = _sample(_random_rows(_rng(0), 15), num=4, start=3, end=10)
    assert all(3 <= i <= 10 for i in idx)


def test_negative_end_index():
    np.random.seed(0)
    rows = _random_rows(_rng(0), 10)
    assert max(_sample(rows, num=3, start=0, end=-1)[0]) <= 9
    assert max(_sample(rows, num=3, start=0, end=-3)[0]) <= 7


def test_num_one_start_equals_end_fast_path():
    idx, dist = _sample(_random_rows(_rng(0), 6), num=1, start=3, end=3)
    assert idx == [3] and dist == [0.0]


def test_num_one_draws_weighted_random_frame():
    # num=1 must go through the weighted draw, not pin to `start`:
    # exactly one in-window index, distances [0.0], and across draws the
    # picks are not always the window start.
    np.random.seed(0)
    rows = _random_rows(_rng(0), 12)
    seen = set()
    for _ in range(40):
        idx, dist = _sample(rows, num=1, start=2, end=9)
        assert len(idx) == 1 and dist == [0.0]
        assert 2 <= idx[0] <= 9
        seen.add(idx[0])
    assert len(seen) > 1 and seen != {2}


def test_num_equals_window_returns_every_frame():
    np.random.seed(0)
    idx, dist = _sample(_random_rows(_rng(0), 6), num=4, start=1, end=4)
    assert idx == [1, 2, 3, 4]
    assert len(dist) == 4 and dist[0] == 0.0


def test_endpoints_not_pinned():
    # Unlike sample_se3_trajectory, endpoints are NOT forced: over many draws
    # at least one must miss an endpoint (P[always pinned] ~ 0).
    np.random.seed(0)
    rows = _random_rows(_rng(0), 20)
    unpinned = 0
    for _ in range(30):
        idx, _ = _sample(rows, num=3, start=0, end=19)
        if idx[0] != 0 or idx[-1] != 19:
            unpinned += 1
    assert unpinned > 0


# --------------------------------------------------------------------------- #
# validation (mirrors the shared _resolve_window contract)
# --------------------------------------------------------------------------- #
def test_asserts_window_at_least_num():
    with pytest.raises(AssertionError):
        _sample(_random_rows(_rng(0), 10), num=5, start=0, end=3)


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
# randomness & determinism (global NumPy RNG)
# --------------------------------------------------------------------------- #
def test_deterministic_under_np_seed():
    rows = _random_rows(_rng(0), 12)
    np.random.seed(123)
    a = _sample(rows, num=5, start=0, end=11)
    np.random.seed(123)
    b = _sample(rows, num=5, start=0, end=11)
    assert a == b


def test_varies_across_draws():
    # Same window, consecutive draws: results must not all be identical
    # (contrast: sample_se3_trajectory is deterministic per window).
    np.random.seed(0)
    rows = _random_rows(_rng(0), 20)
    results = {tuple(_sample(rows, num=5, start=0, end=19)[0]) for _ in range(20)}
    assert len(results) > 1


# --------------------------------------------------------------------------- #
# motion weighting (the core behaviour)
# --------------------------------------------------------------------------- #
def test_motion_frames_dominate_static_ones():
    # 20 identical (static) frames then 10 moving frames: all arc-length lives
    # at indices >= 19, so picks should overwhelmingly land there.
    np.random.seed(0)
    rows = _still_then_move_rows(_rng(0), 20, 10)
    in_motion = total = 0
    for _ in range(300):
        idx, _ = _sample(rows, num=5, start=0, end=29)
        in_motion += sum(1 for i in idx if i >= 19)
        total += len(idx)
    assert in_motion / total > 0.75


def test_gap_diversity_vs_equal_arc():
    # Constant-velocity trajectory: equal-arc sampling would give near-equal
    # inter-sample distances (CV ~ 0); the random draw must produce diverse
    # gaps (CV of pooled spacings well above snapping noise).
    np.random.seed(0)
    rows = _const_velocity_rows(40)
    pooled = []
    for _ in range(100):
        _, dist = _sample(rows, num=5, start=0, end=39)
        pooled.extend(dist[1:])
    pooled = np.asarray(pooled)
    assert pooled.std() / pooled.mean() > 0.35


def test_static_window_uniform_fallback():
    # Zero total arc-length -> uniform draw; exactly num distinct, all
    # distances 0; every frame reachable over many draws.
    np.random.seed(0)
    row = _random_rows(_rng(0), 1)[0]
    rows = [row.copy() for _ in range(9)]
    seen = set()
    for _ in range(200):
        idx, dist = _sample(rows, num=5, start=0, end=8)
        assert len(idx) == 5 and len(set(idx)) == 5
        assert all(d == 0.0 for d in dist)
        seen.update(idx)
    assert seen == set(range(9))


def test_floor_fills_when_moving_frames_fewer_than_num():
    # Motion only near the tail (3 motion-weighted frames: 11, 12, 13) but
    # num=10: without the uniform floor np.random.choice would raise
    # "Fewer non-zero entries in p than size". The floor must fill the
    # remainder from static frames while motion frames still dominate.
    np.random.seed(0)
    rows = _still_then_move_rows(_rng(0), 12, 2)
    idx, _ = _sample(rows, num=10, start=0, end=13)
    assert len(idx) == 10 and len(set(idx)) == 10
    assert sum(1 for i in idx if i in {11, 12, 13}) >= 2


# --------------------------------------------------------------------------- #
# end-to-end against the live TUM sequence API
# --------------------------------------------------------------------------- #
def test_sampler_on_tum_sequence():
    from vggt_omega.datasets.tests.test_sequences import _write_tum_sequence
    from vggt_omega.datasets.sequences.tum import TumSequence

    np.random.seed(0)
    root = tempfile.mkdtemp()
    seq_id = "rgbd_dataset_freiburg1_xyz"
    _write_tum_sequence(root, seq_id, n=8)
    s = TumSequence(root, seq_id)
    sid = s.get_sensors()[0]

    idx, dist = sample_se3_random(s, sid, num=4, start=0, end=7)
    assert len(idx) == 4 and len(set(idx)) == 4
    assert all(0 <= i <= 7 for i in idx)
    assert all(idx[k] < idx[k + 1] for k in range(len(idx) - 1))
    # path arc between kept frames == cum diffs of the real trajectory
    cum = np.concatenate([[0.0], np.cumsum(_gaps(s.get_poses(sid)))])
    for k in range(1, len(idx)):
        np.testing.assert_allclose(dist[k], cum[idx[k]] - cum[idx[k - 1]], atol=1e-9)
