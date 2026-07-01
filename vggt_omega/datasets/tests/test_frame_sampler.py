"""Tests for lightweight random frame-index sampling (sample_frame_indices).

The sampler draws ``num`` *distinct* frame indices uniformly at random from a
``[start, end]`` window and returns them **sorted**, paired with the per-step
index gaps (``distances[k] = idx[k] - idx[k-1]``, ``distances[0] = 0.0``). It
reads only ``seq.get_length(sensor)`` (no poses), so a tiny in-memory ``_LenSeq``
stand-in drives the unit laws; a final test runs it end-to-end on a real
:class:`TumSequence`. Randomness is the global NumPy RNG, seeded per test so the
laws are reproducible.
"""
import tempfile

import numpy as np
import pytest

from vggt_omega.datasets.samplers import sample_frame_indices

N_RANDOM = 30
randomized = pytest.mark.parametrize("seed", range(N_RANDOM))

_SENSOR = 0


class _LenSeq:
    """Minimal BaseSequence-like stand-in: the random index sampler only reads
    ``get_length(sensor)`` (no poses), so this exposes exactly that."""

    def __init__(self, n: int):
        self._n = n

    def get_length(self, sensor_id):
        return self._n


def _sample(n, num, seed=0, **kw):
    """Seed the global RNG then sample, so each law is reproducible."""
    np.random.seed(seed)
    return sample_frame_indices(_LenSeq(n), _SENSOR, num, **kw)


# --------------------------------------------------------------------------- #
# return contract: shape / types / order / index-gap distances
# --------------------------------------------------------------------------- #
@randomized
def test_return_contract(seed):
    idx, dist = _sample(12, num=5, seed=seed, start=0, end=11)
    # equal-length lists of the requested size; native python int / float.
    assert len(idx) == len(dist) == 5
    assert all(isinstance(i, int) for i in idx)
    assert all(isinstance(d, float) for d in dist)
    assert dist[0] == 0.0
    # strictly increasing (distinct + sorted), all within the window.
    assert all(idx[k] < idx[k + 1] for k in range(len(idx) - 1))
    assert all(0 <= i <= 11 for i in idx)
    # the float payload is the per-step index gap.
    for k in range(1, len(idx)):
        assert dist[k] == float(idx[k] - idx[k - 1])


@randomized
def test_count_is_exactly_num(seed):
    # distinct sampling (no dedup) -> exactly num indices on every draw.
    idx, dist = _sample(20, num=7, seed=seed, start=0, end=19)
    assert len(idx) == 7 and len(dist) == 7


@randomized
def test_indices_distinct(seed):
    idx, _ = _sample(20, num=10, seed=seed, start=0, end=19)
    assert len(idx) == len(set(idx))


@randomized
def test_respects_subrange(seed):
    idx, _ = _sample(15, num=4, seed=seed, start=3, end=10)
    assert all(3 <= i <= 10 for i in idx)
    assert idx == sorted(idx)


def test_default_window_is_full():
    # start=0, end=-1 spans the whole sequence; num == len -> every frame.
    idx, _ = _sample(8, num=8, seed=0)
    assert idx == list(range(8))


def test_negative_end_index():
    # end=-1 -> n-1; end=-3 -> n-3. num == window pins the picks to the full range.
    idx, _ = _sample(10, num=10, seed=0, start=0, end=-1)
    assert idx[-1] == 9 and idx == list(range(10))
    idx, _ = _sample(10, num=8, seed=0, start=0, end=-3)
    assert idx[-1] == 7 and idx == list(range(8))


def test_negative_start_index():
    # start=-4 -> n-4 = 6; window [6, 9], num == window -> the full range.
    idx, _ = _sample(10, num=4, seed=0, start=-4, end=-1)
    assert idx == [6, 7, 8, 9]


def test_num_one_returns_start():
    idx, dist = _sample(6, num=1, seed=0, start=2, end=5)
    assert idx == [2] and dist == [0.0]


def test_num_equals_window_returns_every_frame():
    # drawing window-many distinct picks from a window -> the whole window, sorted.
    idx, dist = _sample(6, num=4, seed=3, start=1, end=4)
    assert idx == [1, 2, 3, 4]
    assert dist == [0.0, 1.0, 1.0, 1.0]


# --------------------------------------------------------------------------- #
# randomness: determinism under a fixed seed, variation across seeds, coverage
# --------------------------------------------------------------------------- #
def test_same_seed_is_deterministic():
    a = _sample(50, num=8, seed=123, start=0, end=49)
    b = _sample(50, num=8, seed=123, start=0, end=49)
    assert a == b


def test_different_seeds_vary():
    # the sampler is genuinely random: not every seed yields the same picks.
    results = {tuple(_sample(50, num=8, seed=s, start=0, end=49)[0]) for s in range(20)}
    assert len(results) > 1


def test_uniform_coverage_over_seeds():
    # across many seeds, every frame in the window is eventually selected
    # (P(a frame is missed in one 5-of-10 draw) = 0.5, so ~0.5**200 to miss it).
    seen = set()
    for s in range(200):
        idx, _ = _sample(10, num=5, seed=s, start=0, end=9)
        seen.update(idx)
    assert seen == set(range(10))


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_asserts_window_at_least_num():
    with pytest.raises(AssertionError):
        _sample(10, num=5, seed=0, start=0, end=3)  # window 4 < 5


def test_rejects_nonpositive_num():
    with pytest.raises(ValueError):
        _sample(5, num=0, seed=0)


def test_rejects_end_before_start():
    with pytest.raises(ValueError):
        _sample(6, num=2, seed=0, start=4, end=1)


def test_rejects_out_of_range_window():
    with pytest.raises(ValueError):
        _sample(5, num=2, seed=0, start=0, end=99)


def test_rejects_empty():
    with pytest.raises(ValueError):
        _sample(0, num=1, seed=0)


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
    np.random.seed(0)
    idx, dist = sample_frame_indices(s, sid, num=4, start=0, end=7)
    assert len(idx) == 4 and len(set(idx)) == 4
    assert idx == sorted(idx)
    assert all(0 <= i <= 7 for i in idx)
    assert dist[0] == 0.0
    assert all(dist[k] == float(idx[k] - idx[k - 1]) for k in range(1, 4))
    # index gaps telescope to (last - first).
    assert sum(dist) == float(idx[-1] - idx[0])
