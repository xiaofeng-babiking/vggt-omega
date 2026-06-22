"""Tests for the continuous-time SE(3) trajectory backends.

The backend-agnostic behaviour (fit / interpolate / sample / transform / align)
is written once in :class:`SE3TrajectoryContract` and run against every backend by
subclassing (:class:`TestNumpySE3Trajectory`, :class:`TestTorchSE3Trajectory`),
mirroring ``test_se3_pose.py``. Pose / array outputs are compared in NumPy (via
:func:`_to_np`) so one contract serves both backends.

The cumulative cubic B-spline produced by :meth:`fit` is cross-checked against an
**independent reference** (:func:`bspline_reference`) built directly on
``scipy``'s matrix exp/log — not on the class's own ``exp``/``log`` — so the test
is a genuine check of the spline math, not a tautology.
"""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as _Rotation

from vggt_omega.datasets.se3_pose import NumpySE3Pose, TorchSE3Pose
from vggt_omega.datasets.se3_trajectory import (
    NumpySE3Trajectory,
    TorchSE3Trajectory,
    _umeyama,
)

N_RANDOM = 30
randomized = pytest.mark.parametrize("seed", range(N_RANDOM))
ATOL = 1e-7


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #

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


def assert_pose_close(a, b, atol=ATOL) -> None:
    qa, qb = _to_np(a.quaternion), _to_np(b.quaternion)
    if np.dot(qa, qb) < 0:  # quaternion double cover
        qb = -qb
    np.testing.assert_allclose(qa, qb, atol=atol)
    np.testing.assert_allclose(_to_np(a.translation), _to_np(b.translation), atol=atol)


# ---- independent cumulative cubic B-spline reference (scipy matrix exp/log) ---- #

_REF_BASIS = (1.0 / 6.0) * np.array(
    [[5, 3, -3, 1], [1, 3, 3, -2], [0, 0, 0, 1]], dtype=float
)


def _exp_se3(xi: np.ndarray) -> np.ndarray:
    rho, phi = xi[:3], xi[3:]
    theta = np.linalg.norm(phi)
    K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
    if theta < 1e-9:
        V = np.eye(3) + 0.5 * K + (K @ K) / 6.0
    else:
        V = (np.eye(3) + (1 - np.cos(theta)) / theta**2 * K
             + (theta - np.sin(theta)) / theta**3 * (K @ K))
    T = np.eye(4)
    T[:3, :3] = _Rotation.from_rotvec(phi).as_matrix()
    T[:3, 3] = V @ rho
    return T


def _log_se3(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    phi = _Rotation.from_matrix(R).as_rotvec()
    theta = np.linalg.norm(phi)
    K = np.array([[0, -phi[2], phi[1]], [phi[2], 0, -phi[0]], [-phi[1], phi[0], 0]])
    if theta < 1e-9:
        Vi = np.eye(3) - 0.5 * K + (K @ K) / 12.0
    else:
        c = 1.0 / theta**2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
        Vi = np.eye(3) - 0.5 * K + c * (K @ K)
    return np.concatenate([Vi @ t, phi])


def bspline_reference(mats, t: float, times=None) -> np.ndarray:
    """Cumulative cubic B-spline on SE(3) via scipy matrices (clamped endpoints).

    Independent re-implementation used to cross-check the class under test.

    Args:
        mats: list of ``(4,4)`` control transforms (the input poses).
        t: normalised query time in ``[0, 1]``.
        times: optional per-pose times (any strictly-increasing units). They are
            rescaled to ``[0, 1]`` and used to warp ``t`` onto the segment grid,
            exactly mirroring how the implementation honours non-uniform spacing.
            ``None`` -> uniform knots.

    Endpoints are triple-padded; returns the ``(4,4)`` transform at ``t``.
    """
    ctrl = [mats[0]] * 2 + list(mats) + [mats[-1]] * 2
    n_seg = len(ctrl) - 3  # = N + 1
    t = float(np.clip(t, 0.0, 1.0))
    if times is None:
        knots_t = np.linspace(0.0, 1.0, len(mats))
    else:
        ts = np.asarray(times, dtype=np.float64)
        knots_t = (ts - ts[0]) / (ts[-1] - ts[0])
    knots_g = np.linspace(0.0, float(n_seg), len(mats))
    g = float(np.interp(t, knots_t, knots_g))
    i = int(np.clip(int(np.floor(g)) + 1, 1, n_seg))
    u = float(np.clip(g - (i - 1), 0.0, 1.0))
    b = _REF_BASIS @ np.array([1, u, u * u, u * u * u])
    T = ctrl[i - 1].copy()
    for j in range(3):
        omega = _log_se3(np.linalg.inv(ctrl[i + j - 1]) @ ctrl[i + j])
        T = T @ _exp_se3(b[j] * omega)
    return T


# ----------------------------------------------------------------------------- #
# contract
# ----------------------------------------------------------------------------- #

class SE3TrajectoryContract:
    Pose = None
    Traj = None

    def random_pose(self, rng):
        return self.Pose.from_quat(random_quat(rng), random_trans(rng))

    def random_traj(self, rng, n=5):
        return self.Traj([self.random_pose(rng) for _ in range(n)])

    # ----- construction ----------------------------------------------------- #

    def test_requires_min_two_poses(self):
        with pytest.raises(ValueError):
            self.Traj([self.Pose.identity()])

    def test_rejects_non_increasing_times(self):
        rng = _rng(0)
        with pytest.raises(ValueError):
            self.Traj([self.random_pose(rng), self.random_pose(rng)], times=[1.0, 1.0])

    @randomized
    def test_len_and_poses(self, seed):
        rng = _rng(seed)
        tr = self.random_traj(rng, n=6)
        assert len(tr) == 6
        assert len(tr.poses) == 6

    def test_times_default_uniform_0_1(self):
        rng = _rng(0)
        tr = self.random_traj(rng, n=5)
        np.testing.assert_allclose(_to_np(tr.times), np.linspace(0, 1, 5), atol=1e-12)

    def test_times_explicit_rescaled_to_0_1(self):
        rng = _rng(0)
        tr = self.Traj([self.random_pose(rng) for _ in range(4)], times=[10, 20, 30, 50])
        np.testing.assert_allclose(_to_np(tr.times)[[0, -1]], [0.0, 1.0], atol=1e-12)

    # ----- fit / interpolate ------------------------------------------------ #

    @randomized
    def test_fit_returns_self_and_is_chainable(self, seed):
        tr = self.random_traj(_rng(seed))
        assert tr.fit() is tr

    @randomized
    def test_interpolate_hits_endpoints(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(5)]
        tr = self.Traj(poses).fit()
        assert_pose_close(tr.interpolate(0.0), poses[0])
        assert_pose_close(tr.interpolate(1.0), poses[-1])

    @randomized
    def test_interpolate_matches_bspline_reference(self, seed):
        # Uniform timestamps, many poses, arbitrary in-range query times (not just
        # the knot grid) — checked against the independent scipy reference.
        rng = _rng(seed)
        n = int(rng.integers(4, 9))
        poses = [self.random_pose(rng) for _ in range(n)]
        tr = self.Traj(poses).fit()
        mats = [_to_np(p.transform_matrix) for p in poses]
        query_ts = np.concatenate([
            np.linspace(0.0, 1.0, 7),                 # includes endpoints + knots
            rng.uniform(0.0, 1.0, size=12),           # arbitrary in-range times
        ])
        for t in query_ts:
            got = _to_np(tr.interpolate(float(t)).transform_matrix)
            exp = bspline_reference(mats, float(t))
            np.testing.assert_allclose(got, exp, atol=ATOL, err_msg=f"t={t}")

    @randomized
    def test_interpolate_nonuniform_timestamps_matches_reference(self, seed):
        # Non-uniform (strictly-increasing, arbitrary-unit) timestamps + arbitrary
        # in-range query times. Verifies both that the implementation honours the
        # spacing and that it agrees with the independent reference everywhere.
        rng = _rng(seed)
        n = int(rng.integers(5, 9))
        poses = [self.random_pose(rng) for _ in range(n)]
        # arbitrary increasing timestamps with uneven gaps, in odd units / offset.
        gaps = rng.uniform(0.05, 3.0, size=n - 1)
        t0 = rng.uniform(-10.0, 10.0)
        times = np.concatenate([[t0], t0 + np.cumsum(gaps)])
        tr = self.Traj(poses, times=times).fit()
        mats = [_to_np(p.transform_matrix) for p in poses]

        # endpoints still interpolated exactly regardless of spacing.
        assert_pose_close(tr.interpolate(0.0), poses[0])
        assert_pose_close(tr.interpolate(1.0), poses[-1])

        query_ts = np.concatenate([
            np.array([0.0, 1.0]),
            rng.uniform(0.0, 1.0, size=15),           # arbitrary in-range times
        ])
        for t in query_ts:
            got = _to_np(tr.interpolate(float(t)).transform_matrix)
            exp = bspline_reference(mats, float(t), times=times)
            np.testing.assert_allclose(got, exp, atol=ATOL, err_msg=f"t={t}")

    @randomized
    def test_interpolate_nonuniform_differs_from_uniform(self, seed):
        # Sanity: non-uniform spacing actually changes the interior curve (otherwise
        # the timestamp handling would be a silent no-op).
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(6)]
        uniform = self.Traj(poses).fit()
        skewed = self.Traj(poses, times=[0.0, 0.05, 0.1, 0.2, 0.6, 1.0]).fit()
        diffs = [
            np.linalg.norm(
                _to_np(uniform.interpolate(t).translation)
                - _to_np(skewed.interpolate(t).translation)
            )
            for t in [0.25, 0.4, 0.55, 0.7]
        ]
        assert max(diffs) > 1e-3  # the two curves are meaningfully different


    @randomized
    def test_interpolate_is_continuous(self, seed):
        # C0: tiny step in t -> tiny step in pose (no segment-boundary jumps).
        rng = _rng(seed)
        tr = self.random_traj(rng, n=6).fit()
        for t in np.linspace(0.05, 0.95, 19):
            a = _to_np(tr.interpolate(float(t)).translation)
            b = _to_np(tr.interpolate(float(t) + 1e-5).translation)
            assert np.linalg.norm(a - b) < 1e-2

    def test_interpolate_clamps_out_of_range(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(4)]
        tr = self.Traj(poses).fit()
        assert_pose_close(tr.interpolate(-1.0), poses[0])
        assert_pose_close(tr.interpolate(2.0), poses[-1])

    @randomized
    def test_interpolate_fits_lazily(self, seed):
        # interpolate without calling fit() first should still work.
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(5)]
        tr = self.Traj(poses)
        assert_pose_close(tr.interpolate(0.0), poses[0])

    @randomized
    def test_constant_trajectory_is_constant(self, seed):
        # All-equal control poses -> the curve is that pose everywhere (the spline
        # adds no spurious motion; every relative tangent Ω is zero).
        rng = _rng(seed)
        p = self.random_pose(rng)
        tr = self.Traj([p, p, p, p]).fit()
        for t in [0.0, 0.2, 0.5, 0.8, 1.0]:
            assert_pose_close(tr.interpolate(t), p)

    @randomized
    def test_pure_translation_path_is_smooth_and_endpoint_exact(self, seed):
        # Same orientation, moving translation: the curve keeps that orientation and
        # interpolates the endpoint translations (no rotation coupling).
        rng = _rng(seed)
        q = random_quat(rng)
        poses = [self.Pose.from_quat(q, random_trans(rng)) for _ in range(5)]
        tr = self.Traj(poses).fit()
        assert_pose_close(tr.interpolate(0.0), poses[0])
        assert_pose_close(tr.interpolate(1.0), poses[-1])
        # orientation is unchanged along the whole path.
        for t in [0.0, 0.3, 0.6, 1.0]:
            np.testing.assert_allclose(
                _to_np(tr.interpolate(t).rotation_matrix),
                _to_np(self.Pose.from_quat(q).rotation_matrix),
                atol=ATOL,
            )

    # ----- sample ----------------------------------------------------------- #

    @randomized
    def test_sample_count_and_endpoints(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(5)]
        tr = self.Traj(poses).fit()
        s = tr.sample(0.0, 1.0, 9)
        assert len(s) == 9
        assert_pose_close(s[0], poses[0])
        assert_pose_close(s[-1], poses[-1])

    @randomized
    def test_sample_matches_interpolate_on_subrange(self, seed):
        # sample(start, end, n) must equal interpolate() at the n evenly spaced
        # times across [start, end], inclusive of both ends.
        rng = _rng(seed)
        tr = self.random_traj(rng, n=6).fit()
        start, end, n = 0.2, 0.85, 7
        s = tr.sample(start, end, n)
        assert len(s) == n
        for k, pose in enumerate(s):
            t = start + (end - start) * k / (n - 1)
            assert_pose_close(pose, tr.interpolate(t))
        assert_pose_close(s[0], tr.interpolate(start))
        assert_pose_close(s[-1], tr.interpolate(end))

    def test_sample_one_returns_start(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(4)]
        tr = self.Traj(poses).fit()
        assert len(tr.sample(0.0, 1.0, 1)) == 1
        # n == 1 returns the pose at `start`.
        assert_pose_close(tr.sample(0.0, 1.0, 1)[0], poses[0])
        assert_pose_close(tr.sample(0.4, 1.0, 1)[0], tr.interpolate(0.4))

    def test_sample_clamps_out_of_range_bounds(self):
        rng = _rng(0)
        poses = [self.random_pose(rng) for _ in range(4)]
        tr = self.Traj(poses).fit()
        s = tr.sample(-1.0, 2.0, 5)  # clamped to [0, 1]
        assert_pose_close(s[0], poses[0])
        assert_pose_close(s[-1], poses[-1])

    def test_sample_rejects_nonpositive(self):
        tr = self.random_traj(_rng(0))
        with pytest.raises(ValueError):
            tr.sample(0.0, 1.0, 0)

    # ----- extrapolate ------------------------------------------------------ #

    @randomized
    def test_extrapolate_in_range_matches_interpolate(self, seed):
        # Inside [0, 1] extrapolate must delegate to interpolate exactly.
        rng = _rng(seed)
        tr = self.random_traj(rng, n=6).fit()
        for t in [0.0, 0.2, 0.5, 0.8, 1.0]:
            assert_pose_close(tr.extrapolate(t), tr.interpolate(t))

    @randomized
    def test_extrapolate_seam_is_continuous(self, seed):
        # The curve is C0 across the seam: a tiny step past an endpoint stays close.
        rng = _rng(seed)
        tr = self.random_traj(rng, n=5).fit()
        for t_seam in (0.0, 1.0):
            a = _to_np(tr.extrapolate(t_seam).translation)
            b = _to_np(tr.extrapolate(t_seam + 1e-4).translation)
            c = _to_np(tr.extrapolate(t_seam - 1e-4).translation)
            assert np.linalg.norm(a - b) < 1e-2
            assert np.linalg.norm(a - c) < 1e-2

    @randomized
    def test_extrapolate_end_is_constant_velocity_screw(self, seed):
        # t = 1 + k * (last normalised gap) must equal poses[-1] ∘ exp(k · ξ_end),
        # i.e. a constant-twist screw continuing the last inter-pose motion.
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(5)]
        tr = self.Traj(poses).fit()
        xi = poses[-1].boxminus(poses[-2])  # ξ_end = log(P_{N-2}⁻¹ P_{N-1})
        gap = 1.0 / (len(poses) - 1)  # last normalised gap (uniform times)
        for k in [0.5, 1.0, 2.0, 3.0]:
            got = tr.extrapolate(1.0 + k * gap)
            expect = poses[-1].boxplus(k * xi)
            assert_pose_close(got, expect)

    @randomized
    def test_extrapolate_start_is_constant_velocity_screw(self, seed):
        # Symmetric backward continuation off the start using ξ_start = log(P_0⁻¹ P_1).
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(5)]
        tr = self.Traj(poses).fit()
        xi = poses[1].boxminus(poses[0])
        gap = 1.0 / (len(poses) - 1)
        for k in [0.5, 1.0, 2.0]:
            got = tr.extrapolate(-k * gap)
            expect = poses[0].boxplus(-k * xi)
            assert_pose_close(got, expect)

    @randomized
    def test_extrapolate_honors_nonuniform_last_gap(self, seed):
        # With non-uniform times, one normalised *last-gap* step past the end equals
        # exactly one ξ_end step (the step is scaled by the real last gap, not 1/N).
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(4)]
        times = [0.0, 1.0, 2.0, 10.0]  # last gap dominates
        tr = self.Traj(poses, times=times).fit()
        last_gap = (10.0 - 2.0) / 10.0  # normalised
        xi = poses[-1].boxminus(poses[-2])
        assert_pose_close(tr.extrapolate(1.0 + last_gap), poses[-1].boxplus(xi))
        assert_pose_close(tr.extrapolate(1.0 + 2.0 * last_gap), poses[-1].boxplus(2.0 * xi))

    @randomized
    def test_extrapolate_constant_trajectory_stays_put(self, seed):
        # All-equal poses -> zero boundary twist -> extrapolation never moves.
        rng = _rng(seed)
        p = self.random_pose(rng)
        tr = self.Traj([p, p, p, p]).fit()
        for t in [-1.0, -0.3, 1.5, 3.0]:
            assert_pose_close(tr.extrapolate(t), p)

    # ----- transform -------------------------------------------------------- #

    @randomized
    def test_transform_rigid_matches_pointwise_compose(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(5)]
        tf = self.random_pose(rng)
        tr = self.Traj(poses).transform(tf)
        for orig, new in zip(poses, tr.poses):
            assert_pose_close(new, tf.compose(orig))

    @randomized
    def test_transform_scale_scales_translation(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(4)]
        scale = float(rng.uniform(0.5, 3.0))
        tf = self.random_pose(rng)
        tr = self.Traj(poses).transform(tf, scale=scale)
        for orig, new in zip(poses, tr.poses):
            expect_t = scale * _to_np(tf.compose(orig).translation)
            np.testing.assert_allclose(_to_np(new.translation), expect_t, atol=ATOL)

    @randomized
    def test_transform_returns_new_trajectory(self, seed):
        rng = _rng(seed)
        tr = self.random_traj(rng)
        tr2 = tr.transform(self.Pose.identity())
        assert tr2 is not tr
        np.testing.assert_allclose(_to_np(tr2.times), _to_np(tr.times), atol=1e-12)

    # ----- align ------------------------------------------------------------ #

    @randomized
    def test_align_recovers_known_rigid_transform(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(6)]
        gt = self.random_pose(rng)
        moved = self.Traj(poses).transform(gt)  # known transform applied
        aligned, tf, scale = moved.align(self.Traj(poses), with_scale=False)
        np.testing.assert_allclose(scale, 1.0, atol=ATOL)
        # aligned positions should match the target positions.
        for got, want in zip(aligned.poses, poses):
            np.testing.assert_allclose(
                _to_np(got.translation), _to_np(want.translation), atol=1e-6
            )

    @randomized
    def test_align_recovers_known_scale(self, seed):
        rng = _rng(seed)
        poses = [self.random_pose(rng) for _ in range(6)]
        gt = self.random_pose(rng)
        scale_gt = float(rng.uniform(1.5, 4.0))
        moved = self.Traj(poses).transform(gt, scale=scale_gt)
        # recover: align moved back onto the originals.
        _, _, scale = moved.align(self.Traj(poses), with_scale=True)
        np.testing.assert_allclose(scale, 1.0 / scale_gt, atol=1e-5)

    def test_align_rejects_length_mismatch(self):
        rng = _rng(0)
        a = self.random_traj(rng, n=5)
        b = self.random_traj(rng, n=4)
        with pytest.raises(ValueError):
            a.align(b)

    def test_repr_contains_classname(self):
        assert self.Traj.__name__ in repr(self.random_traj(_rng(0)))


# ----------------------------------------------------------------------------- #
# backends
# ----------------------------------------------------------------------------- #

class TestNumpySE3Trajectory(SE3TrajectoryContract):
    Pose = NumpySE3Pose
    Traj = NumpySE3Trajectory

    @randomized
    def test_outputs_are_numpy(self, seed):
        tr = self.random_traj(_rng(seed)).fit()
        assert isinstance(tr.interpolate(0.3).translation, np.ndarray)
        assert isinstance(tr.times, np.ndarray)


class TestTorchSE3Trajectory(SE3TrajectoryContract):
    Pose = TorchSE3Pose
    Traj = TorchSE3Trajectory

    @randomized
    def test_outputs_are_tensors(self, seed):
        tr = self.random_traj(_rng(seed)).fit()
        assert isinstance(tr.interpolate(0.3).translation, torch.Tensor)
        assert isinstance(tr.times, torch.Tensor)

    def test_dtype_preserved(self):
        rng = _rng(0)
        poses = [
            TorchSE3Pose.from_quat(
                torch.tensor(random_quat(rng), dtype=torch.float32),
                torch.tensor(random_trans(rng), dtype=torch.float32),
            )
            for _ in range(5)
        ]
        tr = TorchSE3Trajectory(poses).fit()
        assert tr.interpolate(0.4).translation.dtype == torch.float32

    @randomized
    def test_grad_flows_through_interpolate(self, seed):
        rng = _rng(seed)
        q = torch.tensor(random_quat(rng), requires_grad=True)
        t = torch.tensor(random_trans(rng), requires_grad=True)
        head = TorchSE3Pose.from_quat(q, t)
        rest = [self.random_pose(rng) for _ in range(4)]
        tr = TorchSE3Trajectory([head] + rest)
        tr.interpolate(0.1).translation.sum().backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert t.grad is not None and torch.isfinite(t.grad).all()

    @randomized
    def test_grad_flows_through_extrapolate(self, seed):
        # Backward extrapolation (t < 0) uses poses[0] / poses[1]; put the grad head
        # at poses[0] so autograd has a path.
        rng = _rng(seed)
        q = torch.tensor(random_quat(rng), requires_grad=True)
        t = torch.tensor(random_trans(rng), requires_grad=True)
        head = TorchSE3Pose.from_quat(q, t)
        rest = [self.random_pose(rng) for _ in range(4)]
        tr = TorchSE3Trajectory([head] + rest)
        out = tr.extrapolate(-0.4)
        assert isinstance(out.translation, torch.Tensor)
        out.translation.sum().backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert t.grad is not None and torch.isfinite(t.grad).all()


# ----------------------------------------------------------------------------- #
# Umeyama unit tests (backend-free)
# ----------------------------------------------------------------------------- #

@randomized
def test_umeyama_recovers_similarity(seed):
    rng = _rng(seed)
    src = rng.uniform(-5, 5, size=(20, 3))
    R_gt = _Rotation.random(random_state=rng).as_matrix()
    s_gt = float(rng.uniform(0.5, 3.0))
    t_gt = rng.uniform(-2, 2, size=3)
    dst = (s_gt * src @ R_gt.T) + t_gt
    R, t, s = _umeyama(src, dst, with_scale=True)
    np.testing.assert_allclose(R, R_gt, atol=1e-6)
    np.testing.assert_allclose(t, t_gt, atol=1e-5)
    np.testing.assert_allclose(s, s_gt, atol=1e-6)
    np.testing.assert_allclose(s * src @ R.T + t, dst, atol=1e-5)


def test_umeyama_proper_rotation():
    rng = _rng(0)
    src = rng.uniform(-5, 5, size=(20, 3))
    dst = rng.uniform(-5, 5, size=(20, 3))
    R, _, _ = _umeyama(src, dst, with_scale=False)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-6)
