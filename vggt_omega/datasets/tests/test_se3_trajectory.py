"""Conformance + correctness tests for ``*SE3Trajectory`` backends.

A random but *smooth* SE(3) trajectory is generated independently of the class
under test (via ``scipy``), and every :class:`BaseSE3Trajectory` API is exercised
against NumPy / SciPy references. The shared checks live in
:class:`BaseSE3TrajectoryContract`; its name does not start with ``Test`` so
pytest never collects it on its own. One ``Test*`` subclass per backend binds the
concrete class (``traj_cls``) -- add a backend by adding a one-line subclass.
"""

import math

import numpy as np
import pytest
from scipy.linalg import expm, logm
from scipy.spatial.transform import Rotation

from vggt_omega.datasets.utils.se3_trajectory import (
    BaseSE3Trajectory,
    NumpySE3Trajectory,
)


# --------------------------------------------------------------------------- #
# random smooth SE(3) trajectory  (backend-independent)
# --------------------------------------------------------------------------- #
def random_smooth_poses(num=10, seed=0, rot_step=0.2, trans_step=0.3):
    """A smooth random SE(3) path as ``(timestamps[N], poses[N, 7])``.

    ``poses`` rows are ``[qx, qy, qz, qw, tx, ty, tz]`` (unit quaternion scalar-last
    + translation). Smoothness: each frame differs from the previous by a *small*
    incremental rotation (~``rot_step`` rad) and a small translation step, so the
    consecutive body twists are bounded -- well away from the ``theta = pi`` branch
    of the matrix log, which keeps ``log`` / ``interpolate`` numerically clean.
    Timestamps are strictly increasing with non-uniform gaps. Deterministic in
    ``seed`` and built without the class under test.
    """
    rng = np.random.default_rng(seed)
    timestamps = np.cumsum(rng.uniform(0.5, 1.5, size=num))  # strictly increasing
    increments = Rotation.from_rotvec(rng.normal(size=(num, 3)) * rot_step)
    acc = Rotation.identity()
    quats = np.empty((num, 4))
    for k in range(num):
        acc = acc * increments[k]  # accumulate -> smooth orientation path
        quats[k] = acc.as_quat()  # xyzw
    trans = np.cumsum(rng.normal(size=(num, 3)) * trans_step, axis=0)
    return timestamps, np.concatenate([quats, trans], axis=1)


# --- independent SE(3) exp/log references (scipy matrix exp/log) ------------- #
def _hat(xi):
    """``(6,)`` twist ``(rho, phi)`` (translation first) -> ``(4, 4)`` se(3) matrix."""
    rho, phi = xi[:3], xi[3:]
    px, py, pz = phi
    out = np.zeros((4, 4))
    out[:3, :3] = [[0.0, -pz, py], [pz, 0.0, -px], [-py, px, 0.0]]
    out[:3, 3] = rho
    return out


def ref_exp(xi):
    return expm(_hat(xi))


def ref_log(tf):
    mat = logm(tf).real
    return np.array([mat[0, 3], mat[1, 3], mat[2, 3], mat[2, 1], mat[0, 2], mat[1, 0]])


def _little_adjoint(xi):
    """6x6 ``ad_xi`` for the ``(rho, phi)`` ordering (used for the J_l series)."""

    def sk(v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)

    rho, phi = xi[:3], xi[3:]
    ad = np.zeros((6, 6))
    ad[:3, :3] = sk(phi)
    ad[:3, 3:] = sk(rho)
    ad[3:, 3:] = sk(phi)
    return ad


# --------------------------------------------------------------------------- #
# shared contract — runs against every BaseSE3Trajectory backend
# --------------------------------------------------------------------------- #
class BaseSE3TrajectoryContract:
    """API laws every :class:`BaseSE3Trajectory` subclass must satisfy.

    Subclasses set :attr:`traj_cls` to the concrete backend. The name does not
    start with ``Test`` so pytest does not collect this mixin directly.
    """

    traj_cls = None  # set by the concrete Test* subclass

    # -- factory --------------------------------------------------------- #
    def make(self, num=10, seed=0, source="cam", target="world"):
        """Build a trajectory of ``traj_cls`` from a random smooth path.

        Returns ``(trajectory, timestamps, poses)`` for reference comparisons.
        """
        ts, poses = random_smooth_poses(num=num, seed=seed)
        return self.traj_cls(ts, poses, source, target), ts, poses

    # ===== construction / sequence protocol ============================= #
    def test_len(self):
        traj, _, _ = self.make(num=7)
        assert len(traj) == 7

    def test_identity(self):
        ident = self.traj_cls.identity(4, timestamps=[0.0, 1.0, 2.0, 3.0])
        assert len(ident) == 4
        np.testing.assert_allclose(
            ident.transform_matrix(), np.broadcast_to(np.eye(4), (4, 4, 4))
        )
        assert len(self.traj_cls.identity()) == 1  # default num == 1

    def test_getitem(self):
        traj, ts, _ = self.make(num=8)
        tf = traj.transform_matrix()
        # slice -> sub-trajectory
        np.testing.assert_allclose(traj[2:5].transform_matrix(), tf[2:5])
        np.testing.assert_allclose(traj[2:5]._tstamps, ts[2:5])
        # int -> length-1 trajectory (batch axis kept)
        sub = traj[3]
        assert len(sub) == 1
        np.testing.assert_allclose(sub.transform_matrix()[0], tf[3])
        # fancy index (reorder + repeat)
        np.testing.assert_allclose(
            traj[[5, 0, 3, 0]].transform_matrix(), tf[[5, 0, 3, 0]]
        )
        # boolean mask
        mask = np.zeros(8, dtype=bool)
        mask[::2] = True
        np.testing.assert_allclose(traj[mask].transform_matrix(), tf[mask])
        np.testing.assert_allclose(traj[mask]._tstamps, ts[mask])

    def test_fit_not_implemented(self):
        traj, _, _ = self.make()
        with pytest.raises(NotImplementedError):
            traj.fit()

    # ===== representation converters ==================================== #
    def test_transform_matrix(self):
        traj, _, poses = self.make()
        n = len(traj)
        tf = traj.transform_matrix()
        assert tf.shape == (n, 4, 4)
        np.testing.assert_allclose(tf[:, 3, :], np.tile([0, 0, 0, 1.0], (n, 1)))
        rot = tf[:, :3, :3]
        np.testing.assert_allclose(  # orthonormal
            np.einsum("nij,nkj->nik", rot, rot),
            np.broadcast_to(np.eye(3), (n, 3, 3)),
            atol=1e-9,
        )
        np.testing.assert_allclose(np.linalg.det(rot), np.ones(n), atol=1e-9)
        np.testing.assert_allclose(rot, Rotation.from_quat(poses[:, :4]).as_matrix())
        np.testing.assert_allclose(tf[:, :3, 3], poses[:, 4:7])

    def test_se3_group_is_transform_matrix(self):
        traj, _, _ = self.make()
        np.testing.assert_allclose(traj.se3_group(), traj.transform_matrix())

    def test_rotation_matrix(self):
        traj, _, poses = self.make()
        np.testing.assert_allclose(
            traj.rotation_matrix(), Rotation.from_quat(poses[:, :4]).as_matrix()
        )

    def test_rotation_vector(self):
        traj, _, _ = self.make()
        rv = traj.rotation_vector()
        assert rv.shape == (len(traj), 3)
        np.testing.assert_allclose(
            Rotation.from_rotvec(rv).as_matrix(), traj.rotation_matrix(), atol=1e-9
        )

    def test_quaternion(self):
        traj, _, _ = self.make()
        quat = traj.quaternion()  # xyzw
        np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1.0)  # unit
        np.testing.assert_allclose(
            Rotation.from_quat(quat).as_matrix(), traj.rotation_matrix()
        )
        np.testing.assert_allclose(
            traj.quaternion(scalar_last=False), quat[:, [3, 0, 1, 2]]
        )

    def test_translation(self):
        traj, _, poses = self.make()
        np.testing.assert_allclose(traj.translation(), poses[:, 4:7])

    def test_euler_angles(self):
        traj, _, _ = self.make()
        deg = traj.euler_angles("ZYX", degress=True)
        np.testing.assert_allclose(
            Rotation.from_euler("ZYX", deg, degrees=True).as_matrix(),
            traj.rotation_matrix(),
            atol=1e-8,
        )
        rad = traj.euler_angles("xyz", degress=False)
        np.testing.assert_allclose(
            Rotation.from_euler("xyz", rad, degrees=False).as_matrix(),
            traj.rotation_matrix(),
            atol=1e-8,
        )

    # ===== Lie-group utilities ========================================== #
    def test_exp_log_roundtrip(self):
        traj, _, _ = self.make()
        tf = traj.transform_matrix()
        np.testing.assert_allclose(
            self.traj_cls.exp(self.traj_cls.log(tf)), tf, atol=1e-9
        )
        xi = traj.se3_algebra()
        np.testing.assert_allclose(
            self.traj_cls.log(self.traj_cls.exp(xi)), xi, atol=1e-9
        )

    def test_exp_log_vs_reference(self):
        """Cross-check exp/log (and the (rho, phi) ordering) vs scipy expm/logm."""
        traj, _, _ = self.make()
        tf = traj.transform_matrix()
        xi = self.traj_cls.log(tf)
        ref_xi = np.array([ref_log(tf[i]) for i in range(len(traj))])
        np.testing.assert_allclose(xi, ref_xi, atol=1e-8)
        back = self.traj_cls.exp(xi)
        ref_tf = np.array([ref_exp(xi[i]) for i in range(len(traj))])
        np.testing.assert_allclose(back, ref_tf, atol=1e-8)

    def test_se3_algebra(self):
        traj, _, _ = self.make()
        xi = traj.se3_algebra()
        assert xi.shape == (len(traj), 6)
        np.testing.assert_allclose(
            self.traj_cls.exp(xi), traj.transform_matrix(), atol=1e-9
        )
        np.testing.assert_allclose(xi, self.traj_cls.log(traj.transform_matrix()))

    def test_norm(self):
        traj, _, _ = self.make()
        norm = traj.norm()
        assert norm.shape == (len(traj),)
        np.testing.assert_allclose(norm, np.linalg.norm(traj.se3_algebra(), axis=1))

    def test_left_jacobian(self):
        traj, _, _ = self.make()
        xi = traj.se3_algebra()
        jac = self.traj_cls.left_jacobian(xi)
        assert jac.shape == (len(traj), 6, 6)
        for i in range(len(traj)):  # J_l = sum_k ad^k / (k+1)!
            ad = _little_adjoint(xi[i])
            series = sum(
                (
                    np.linalg.matrix_power(ad, k) / math.factorial(k + 1)
                    for k in range(20)
                ),
                np.zeros((6, 6)),
            )
            np.testing.assert_allclose(jac[i], series, atol=1e-7)
        # accepts a group element too (logged first)
        np.testing.assert_allclose(
            self.traj_cls.left_jacobian(traj.transform_matrix()), jac, atol=1e-9
        )

    def test_batched_ndim_kept(self):
        """A single element keeps the batch axis ([1, ...]), never squeezed."""
        traj, _, _ = self.make()
        tf = traj.transform_matrix()
        assert self.traj_cls.log(tf[0]).shape == (1, 6)
        assert self.traj_cls.exp(traj.se3_algebra()[0]).shape == (1, 4, 4)
        assert self.traj_cls.left_jacobian(tf[0]).shape == (1, 6, 6)
        assert self.traj_cls.left_jacobian(np.zeros(6)).shape == (1, 6, 6)
        solo, _, _ = self.make(num=1)
        assert solo.transform_matrix().shape == (1, 4, 4)
        assert solo.se3_algebra().shape == (1, 6)
        assert solo.norm().shape == (1,)

    def test_adjoint(self):
        traj, _, _ = self.make()
        tf = traj.transform_matrix()
        ad = traj.adjoint()
        assert ad.shape == (len(traj), 6, 6)
        # Ad_T @ xi == log(T exp(xi) T^-1)
        xi = np.random.default_rng(11).normal(size=(len(traj), 6)) * 0.1
        lhs = np.einsum("nij,nj->ni", ad, xi)
        rhs = self.traj_cls.log(tf @ self.traj_cls.exp(xi) @ np.linalg.inv(tf))
        np.testing.assert_allclose(lhs, rhs, atol=1e-7)

    # ===== SE(3) group operations ======================================= #
    def test_inverse(self):
        traj, _, _ = self.make()
        n = len(traj)
        inv = traj.inverse()
        np.testing.assert_allclose(
            traj.transform_matrix() @ inv.transform_matrix(),
            np.broadcast_to(np.eye(4), (n, 4, 4)),
            atol=1e-9,
        )
        assert inv._src == traj._dst and inv._dst == traj._src  # frames swapped

    def test_compose(self):
        a, _, _ = self.make(seed=1)
        b, _, _ = self.make(seed=2)
        np.testing.assert_allclose(
            a.compose(b).transform_matrix(),
            a.transform_matrix() @ b.transform_matrix(),
            atol=1e-9,
        )
        with pytest.raises(ValueError):
            a.compose(self.make(num=3, seed=3)[0])  # length mismatch

    def test_relative(self):
        a, _, _ = self.make(seed=1)
        b, _, _ = self.make(seed=2)
        rel = a.relative(b)  # satisfies b.compose(rel) == a
        np.testing.assert_allclose(
            b.compose(rel).transform_matrix(), a.transform_matrix(), atol=1e-9
        )

    def test_boxplus_boxminus(self):
        a, _, _ = self.make(seed=1)
        b, _, _ = self.make(seed=2)
        tw = np.random.default_rng(5).normal(size=(len(a), 6)) * 0.2
        # (a ⊞ tw) ⊟ a == tw
        np.testing.assert_allclose(a.boxplus(tw).boxminus(a), tw, atol=1e-9)
        # b ⊞ (a ⊟ b) == a
        np.testing.assert_allclose(
            b.boxplus(a.boxminus(b)).transform_matrix(),
            a.transform_matrix(),
            atol=1e-9,
        )
        # boxminus == log(inv(b) @ a)
        np.testing.assert_allclose(
            a.boxminus(b),
            self.traj_cls.log(
                np.linalg.inv(b.transform_matrix()) @ a.transform_matrix()
            ),
            atol=1e-9,
        )

    def test_consecutive_twist(self):
        traj, _, _ = self.make()
        tf = traj.transform_matrix()
        twist = traj.consecutive_twist()
        assert twist.shape == (len(traj) - 1, 6)
        np.testing.assert_allclose(
            twist, self.traj_cls.log(np.linalg.inv(tf[:-1]) @ tf[1:]), atol=1e-9
        )
        # consistent with the manifold ⊟ between adjacent frames
        np.testing.assert_allclose(twist, traj[1:].boxminus(traj[:-1]), atol=1e-9)

    def test_apply(self):
        traj, _, _ = self.make()
        n = len(traj)
        rot, trans = traj.rotation_matrix(), traj.translation()
        # single point -> broadcast to every frame
        p = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(traj.apply(p), rot @ p + trans)
        # one point per frame (paired)
        pn = np.random.default_rng(7).normal(size=(n, 3))
        np.testing.assert_allclose(
            traj.apply(pn), np.einsum("nij,nj->ni", rot, pn) + trans
        )
        # (M, 3) cloud -> broadcast to every frame
        pm = np.random.default_rng(8).normal(size=(4, 3))
        np.testing.assert_allclose(
            traj.apply(pm), np.einsum("nij,mj->nmi", rot, pm) + trans[:, None, :]
        )
        with pytest.raises(ValueError):
            traj.apply(np.zeros((2, 4)))

    # ===== temporal / registration ====================================== #
    def test_interpolate_at_samples(self):
        traj, ts, _ = self.make()
        np.testing.assert_allclose(
            traj.interpolate(ts).transform_matrix(), traj.transform_matrix(), atol=1e-9
        )
        np.testing.assert_allclose(traj.interpolate(ts)._tstamps, ts)

    def test_interpolate_midpoints_valid(self):
        traj, ts, _ = self.make()
        mid = (ts[:-1] + ts[1:]) / 2.0
        gi = traj.interpolate(mid).transform_matrix()
        assert gi.shape == (len(mid), 4, 4)
        rot = gi[:, :3, :3]
        np.testing.assert_allclose(
            np.einsum("nij,nkj->nik", rot, rot),
            np.broadcast_to(np.eye(3), (len(mid), 3, 3)),
            atol=1e-9,
        )

    def test_interpolate_pure_translation_is_lerp(self):
        line = self.traj_cls(
            np.array([0.0, 2.0]),
            np.array([[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 4, 0, 0]], dtype=float),
            "a",
            "b",
        )
        mid = line.interpolate(np.array([1.0])).transform_matrix()[0]
        np.testing.assert_allclose(mid[:3, 3], [2.0, 0.0, 0.0], atol=1e-9)

    def test_interpolate_errors(self):
        traj, ts, poses = self.make()
        with pytest.raises(ValueError):
            traj.interpolate([ts[0] - 1.0])  # below range
        with pytest.raises(ValueError):
            traj.interpolate([ts[-1] + 1.0])  # above range
        with pytest.raises(ValueError):  # no time axis
            self.traj_cls(None, poses, "a", "b").interpolate([0.5])

    def test_align_recovers_rigid(self):
        gt, ts, _ = self.make(seed=1)
        rng = np.random.default_rng(9)
        rot0 = Rotation.from_rotvec(rng.normal(size=3) * 0.3).as_matrix()
        trans0 = rng.normal(size=3)
        target_pos = gt.translation()
        moved_pos = (target_pos - trans0) @ rot0  # rot0 @ moved + trans0 == target
        moved = self.traj_cls(
            ts, np.concatenate([gt.quaternion(), moved_pos], axis=1), "a", "b"
        )
        aligned = moved.align(gt)
        np.testing.assert_allclose(aligned.translation(), target_pos, atol=1e-6)


# --------------------------------------------------------------------------- #
# concrete backends — one subclass per *SE3Trajectory implementation
# --------------------------------------------------------------------------- #
class TestNumpySE3Trajectory(BaseSE3TrajectoryContract):
    traj_cls = NumpySE3Trajectory


def test_numpy_is_base_subclass():
    assert issubclass(NumpySE3Trajectory, BaseSE3Trajectory)
