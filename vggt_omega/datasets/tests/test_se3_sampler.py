"""Tests for SE(3) arc-length frame sampling (sample_se3_trajectory).

The sampler is backend-agnostic (works on NumPy or torch poses); the same checks
run against both backends via a small contract mixin, mirroring test_se3_pose.py.
"""
import numpy as np
import pytest
import torch

from vggt_omega.datasets.se3_pose import NumpySE3Pose, TorchSE3Pose
from vggt_omega.datasets.samplers import sample_se3_trajectory

N_RANDOM = 30
randomized = pytest.mark.parametrize("seed", range(N_RANDOM))

_SENSOR = 0


class _PoseSeq:
    """Minimal BaseSequence-like pose source for the sampler: exposes the two
    methods the sampler uses -- get_length(sensor) and get_pose(sensor, k) -- over
    a plain in-memory list of poses. Records which frame ids were read so tests can
    assert the sampler only touches the [start, end] window."""

    def __init__(self, poses):
        self._poses = list(poses)
        self.read_ids = []

    def get_length(self, sensor_id):
        return len(self._poses)

    def get_pose(self, sensor_id, frame_id):
        self.read_ids.append(int(frame_id))
        return self._poses[int(frame_id)]


def _sample(poses, num, **kw):
    """Adapt the historical list-based test calls to the (seq, sensor, num, ...)
    API; returns the sampler result. Use ``_sample_seq`` when you also need the
    sequence to inspect which frames were read."""
    return sample_se3_trajectory(_PoseSeq(poses), _SENSOR, num, **kw)


def _sample_seq(poses, num, **kw):
    seq = _PoseSeq(poses)
    return seq, sample_se3_trajectory(seq, _SENSOR, num, **kw)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def random_quat(rng) -> np.ndarray:
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def random_trans(rng, scale=5.0) -> np.ndarray:
    return rng.uniform(-scale, scale, size=3)


def twist_norm(a, b) -> float:
    """SE(3) gap ‖log(b⁻¹ a)‖ between two poses (the per-frame motion metric)."""
    return float(np.linalg.norm(_to_np(a.boxminus(b))))


class SamplerContract:
    Pose = None

    def random_pose(self, rng):
        return self.Pose.from_quat(random_quat(rng), random_trans(rng))

    # ----- shape / endpoints / monotonicity / return contract --------------- #

    @randomized
    def test_return_contract(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(12)]
        idx, dist = _sample(poses, num=5, start=0, end=11)
        # equal-length lists; distances[0] == 0.0
        assert len(idx) == len(dist)
        assert dist[0] == 0.0
        assert all(isinstance(i, int) for i in idx)
        assert all(isinstance(d, float) for d in dist)
        # endpoints pinned; strictly increasing (duplicates removed)
        assert idx[0] == 0 and idx[-1] == 11
        assert all(idx[k] < idx[k + 1] for k in range(len(idx) - 1))
        assert all(0 <= i < len(poses) for i in idx)
        # distances are the path arc-length (sum of per-frame gaps) between kept idx
        gaps = [twist_norm(poses[k], poses[k - 1]) for k in range(1, len(poses))]
        cum = np.concatenate([[0.0], np.cumsum(gaps)])
        for k in range(1, len(idx)):
            np.testing.assert_allclose(dist[k], cum[idx[k]] - cum[idx[k - 1]], atol=1e-9)

    @randomized
    def test_count_at_most_num(self, seed):
        # de-duplication can only shrink the result, never exceed num.
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(12)]
        idx, dist = _sample(poses, num=5, start=0, end=11)
        assert 2 <= len(idx) <= 5
        assert len(dist) == len(idx)

    @randomized
    def test_respects_subrange(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(15)]
        idx, _ = _sample(poses, num=4, start=3, end=10)
        assert idx[0] == 3 and idx[-1] == 10
        assert all(3 <= i <= 10 for i in idx)

    def test_default_window_is_full(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(8)]
        idx, _ = _sample(poses, num=4)  # start=0, end=-1
        assert idx[0] == 0 and idx[-1] == 7

    def test_negative_end_index(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(10)]
        assert _sample(poses, num=3, start=0, end=-1)[0][-1] == 9
        assert _sample(poses, num=3, start=0, end=-3)[0][-1] == 7

    def test_num_one_returns_start(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(6)]
        idx, dist = _sample(poses, num=1, start=2, end=5)
        assert idx == [2] and dist == [0.0]

    def test_num_equals_window_returns_every_frame(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(6)]
        idx, dist = _sample(poses, num=4, start=1, end=4)
        assert idx == [1, 2, 3, 4]
        assert len(dist) == 4 and dist[0] == 0.0

    # ----- validation -------------------------------------------------------- #

    def test_asserts_window_at_least_num(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(10)]
        with pytest.raises(AssertionError):
            _sample(poses, num=5, start=0, end=3)  # window 4 < 5

    def test_rejects_nonpositive_num(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(5)]
        with pytest.raises(ValueError):
            _sample(poses, num=0)

    def test_rejects_end_before_start(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(6)]
        with pytest.raises(ValueError):
            _sample(poses, num=2, start=4, end=1)

    def test_rejects_out_of_range_window(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(5)]
        with pytest.raises(ValueError):
            _sample(poses, num=2, start=0, end=99)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            _sample([], num=1)

    # ----- arc-length adaptivity (the core behaviour) ----------------------- #

    @randomized
    def test_skips_small_motion_frames(self, seed):
        # 4 near-static frames then 5 moving frames: at most one interior sample
        # should fall in the static region; the rest land on the moving stretch.
        rng = _rng(seed)
        still = self.random_pose(rng)
        movers = [self.random_pose(rng) for _ in range(5)]
        poses = [still, still, still, still] + movers  # frames 0..3 static
        idx, _ = _sample(poses, num=5, start=0, end=len(poses) - 1)
        interior_in_static = sum(1 for i in idx[1:-1] if i < 4)
        assert interior_in_static <= 1

    @randomized
    def test_equal_arclength_spacing(self, seed):
        # Selected frames are ~equally spaced in cumulative SE(3) arc-length.
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(16)]
        num = 5
        idx, dist = _sample(poses, num=num, start=0, end=len(poses) - 1)
        # the returned distances ARE the inter-sample arc-lengths (skip dist[0]==0).
        seg = dist[1:]
        ideal = sum(seg) / len(seg)
        assert all(s > 0 for s in seg)
        # snapping to discrete frames is approximate -> loose tolerance.
        assert max(abs(s - ideal) for s in seg) < 0.6 * ideal

    @randomized
    def test_dedup_removes_duplicate_snaps(self, seed):
        # A long static head packed with many targets must collapse to unique
        # indices: no repeats, lengths still match.
        rng = _rng(seed)
        still = self.random_pose(rng)
        poses = [still] * 8 + [self.random_pose(rng) for _ in range(2)]
        idx, dist = _sample(poses, num=8, start=0, end=len(poses) - 1)
        assert len(idx) == len(set(idx))  # no duplicates
        assert len(idx) == len(dist)
        assert idx[0] == 0 and idx[-1] == len(poses) - 1

    def test_no_motion_spreads_by_index(self):
        # All-equal poses -> zero arc-length -> even index spacing, all gaps zero.
        rng = _rng(0)
        p = self.random_pose(rng)
        idx, dist = _sample([p] * 9, num=5, start=0, end=8)
        assert idx == [0, 2, 4, 6, 8]
        assert dist == [0.0, 0.0, 0.0, 0.0, 0.0]

    @randomized
    def test_deterministic_for_fixed_window(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(12)]
        a = _sample(poses, num=5, start=1, end=10)
        b = _sample(poses, num=5, start=1, end=10)
        assert a == b

    def test_reads_only_window_frames(self):
        # The sampler must touch ONLY frames in [start, end] (the perf fix):
        # frames outside the window are never read from the sequence.
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(50)]
        seq, _ = _sample_seq(poses, num=4, start=10, end=19)
        assert seq.read_ids, "sampler read no poses"
        assert min(seq.read_ids) >= 10 and max(seq.read_ids) <= 19
        assert set(seq.read_ids) == set(range(10, 20))  # every window frame, none outside


class TestNumpySampler(SamplerContract):
    Pose = NumpySE3Pose


class TestTorchSampler(SamplerContract):
    Pose = TorchSE3Pose
