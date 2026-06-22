"""Randomized property tests for the SE(3) pose backends.

The SE(3) group/Lie laws are written once, in the backend-agnostic
:class:`SE3PoseContract`, and run against every backend by subclassing it
(:class:`TestNumpySE3Pose`, :class:`TestTorchSE3Pose`). Each law is parametrized
over many random poses, so the laws and round-trips are stressed across the whole
manifold. Pose outputs are compared in NumPy (via :func:`_to_np`) so one contract
serves both backends; raw random inputs are float64 NumPy arrays — fed to
``Pose.from_*`` they yield float64 poses on either backend, so the tight tolerance
holds. Each backend then adds the guarantees the other can't make (NumPy: ndarray
outputs; torch: tensor outputs, dtype/device preservation, autograd, Euler-vs-scipy).

``SE3PoseContract`` does not match ``Test*`` so it is not collected on its own —
only the two backend subclasses are.
"""
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as _Rotation, Slerp as _Slerp

from vggt_omega.datasets.se3_pose import NumpySE3Pose, TorchSE3Pose

# Each randomized law runs over this many independent random cases.
N_RANDOM = 50
randomized = pytest.mark.parametrize("seed", range(N_RANDOM))

ATOL = 1e-7  # above the ~1e-10 numerical floor, with slack for cross-backend float64


# ----------------------------------------------------------------------------- #
# random generators & helpers
# ----------------------------------------------------------------------------- #

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _to_np(x) -> np.ndarray:
    """Pose output (NumPy array or torch tensor) -> NumPy for comparison."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform on SO(3): a normalized 4D Gaussian is uniform on S^3 (xyzw)."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def random_trans(rng: np.random.Generator, scale: float = 5.0) -> np.ndarray:
    return rng.uniform(-scale, scale, size=3)


def random_rot_vec(rng: np.random.Generator, max_angle: float = np.pi) -> np.ndarray:
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    return axis * rng.uniform(0.0, max_angle)


def random_tangent(rng: np.random.Generator) -> np.ndarray:
    """Twist ``(ρ, φ)`` with ``‖φ‖ < π`` so ``exp`` is injective (``log`` inverts)."""
    return np.concatenate([random_trans(rng), random_rot_vec(rng, np.pi - 1e-6)])


def random_points(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.uniform(-10.0, 10.0, size=shape)


def assert_pose_close(a, b, atol: float = ATOL) -> None:
    """Equal poses, accounting for the quaternion double cover (q ≡ -q)."""
    qa, qb = _to_np(a.quaternion), _to_np(b.quaternion)
    if np.dot(qa, qb) < 0:
        qb = -qb
    np.testing.assert_allclose(qa, qb, atol=atol)
    np.testing.assert_allclose(_to_np(a.translation), _to_np(b.translation), atol=atol)


def interpolate_groundtruth(a, b, t: float):
    """Reference SE(3) geodesic interpolation, independent of the class under test.

    This is the constant-twist *screw motion* ``a ∘ exp(t·log(a⁻¹∘b))`` that
    :meth:`BaseSE3Pose.interpolate` implements, but reconstructed from first
    principles with ``scipy`` so it cross-checks the class's own ``exp``/``log``:

    * **rotation** = quaternion slerp (``scipy.Slerp``, shortest arc) — constant
      angular velocity from ``a`` to ``b``;
    * **translation** = the screw path, *not* a straight-line lerp. The relative
      translation is mapped into twist coordinates ``ρ = V(φ)⁻¹ · t_rel`` and
      re-integrated at fraction ``t`` through ``V(t·φ)`` (the SO(3) left Jacobian,
      the time-averaged rotation), then carried back into the world frame:
      ``t(t) = R_a · V(t·φ) · (t·ρ) + t_a``.

    Returns ``(quat_xyzw, trans)`` as float64 NumPy arrays.
    """
    qa, qb = _to_np(a.quaternion), _to_np(b.quaternion)
    ta, tb = _to_np(a.translation), _to_np(b.translation)
    Ra = _Rotation.from_quat(qa).as_matrix()

    # --- rotation: shortest-arc quaternion slerp ---
    if np.dot(qa, qb) < 0:  # same hemisphere -> short arc
        qb = -qb
    slerp = _Slerp([0.0, 1.0], _Rotation.from_quat(np.stack([qa, qb])))
    quat = slerp([t])[0].as_quat()  # scalar-last xyzw

    # --- translation: SE(3) screw path via the SO(3) left Jacobian V ---
    R_rel = Ra.T @ _Rotation.from_quat(qb).as_matrix()
    phi = _Rotation.from_matrix(R_rel).as_rotvec()          # relative rotation axis-angle
    t_rel = Ra.T @ (tb - ta)                                # relative translation, in a's frame
    rho = np.linalg.solve(_left_jacobian_so3(phi), t_rel)   # ρ = V(φ)⁻¹ · t_rel
    trans = Ra @ (_left_jacobian_so3(t * phi) @ (t * rho)) + ta
    return quat, trans


def _left_jacobian_so3(phi: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian ``V(φ)``, computed independently of the class under test."""
    theta = float(np.linalg.norm(phi))
    K = np.array([[0.0, -phi[2], phi[1]],
                  [phi[2], 0.0, -phi[0]],
                  [-phi[1], phi[0], 0.0]])
    if theta < 1e-10:  # Taylor: I + 1/2 K + 1/6 K^2
        return np.eye(3) + 0.5 * K + (K @ K) / 6.0
    a = (1.0 - np.cos(theta)) / theta**2
    b = (theta - np.sin(theta)) / theta**3
    return np.eye(3) + a * K + b * (K @ K)



# ----------------------------------------------------------------------------- #
# SE3PoseContract — the laws, run against every backend via subclassing
# ----------------------------------------------------------------------------- #

class SE3PoseContract:
    """Randomized SE(3) laws. Subclass and set ``Pose``."""

    Pose = None  # set by subclass

    def random_pose(self, rng: np.random.Generator):
        return self.Pose.from_quat(random_quat(rng), random_trans(rng))

    # ----- construction & accessors ----------------------------------------- #

    @randomized
    def test_identity_is_noop(self, seed):
        rng = _rng(seed)
        ident = self.Pose.identity()
        np.testing.assert_allclose(_to_np(ident.quaternion), [0, 0, 0, 1], atol=1e-12)
        pts = random_points(rng, (int(rng.integers(1, 8)), 3))
        np.testing.assert_allclose(_to_np(ident.apply(pts)), pts, atol=ATOL)

    @randomized
    def test_from_quat_normalizes(self, seed):
        rng = _rng(seed)
        q = random_quat(rng)
        pose = self.Pose.from_quat(q * rng.uniform(0.1, 10.0))  # off the unit sphere
        np.testing.assert_allclose(np.linalg.norm(_to_np(pose.quaternion)), 1.0, atol=1e-12)
        assert_pose_close(pose, self.Pose.from_quat(q))

    @randomized
    def test_from_quat_wxyz_is_reordered(self, seed):
        q_xyzw = random_quat(_rng(seed))
        q_wxyz = q_xyzw[[3, 0, 1, 2]]
        assert_pose_close(
            self.Pose.from_quat(q_wxyz, scalar_last=False),
            self.Pose.from_quat(q_xyzw),
        )

    @randomized
    def test_from_quat_default_zero_translation(self, seed):
        pose = self.Pose.from_quat(random_quat(_rng(seed)))
        np.testing.assert_allclose(_to_np(pose.translation), [0, 0, 0], atol=1e-12)

    @randomized
    def test_apply_is_right_handed_about_z(self, seed):
        # +θ about z maps e_x -> (cosθ, sinθ, 0): pins right-handedness, random angle.
        theta = _rng(seed).uniform(-np.pi, np.pi)
        pose = self.Pose.from_rot_vec([0, 0, theta])
        np.testing.assert_allclose(
            _to_np(pose.apply([1, 0, 0])), [np.cos(theta), np.sin(theta), 0], atol=ATOL
        )

    @randomized
    def test_apply_matches_rotation_matrix_form(self, seed):
        rng = _rng(seed)
        pose = self.random_pose(rng)
        pts = random_points(rng, (int(rng.integers(1, 8)), 3))
        R, t = _to_np(pose.rotation_matrix), _to_np(pose.translation)
        np.testing.assert_allclose(_to_np(pose.apply(pts)), pts @ R.T + t, atol=ATOL)

    @randomized
    def test_apply_preserves_point_array_shape(self, seed):
        rng = _rng(seed)
        pose = self.random_pose(rng)
        for shape in [(3,), (5, 3), (2, 4, 3)]:
            assert tuple(_to_np(pose.apply(random_points(rng, shape))).shape) == shape

    @randomized
    def test_rotation_matrix_is_special_orthogonal(self, seed):
        R = _to_np(self.random_pose(_rng(seed)).rotation_matrix)
        assert R.shape == (3, 3)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=ATOL)
        assert np.isclose(np.linalg.det(R), 1.0, atol=ATOL)

    @randomized
    def test_transform_matrix_blocks(self, seed):
        pose = self.random_pose(_rng(seed))
        T = _to_np(pose.transform_matrix)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3], _to_np(pose.rotation_matrix), atol=1e-12)
        np.testing.assert_allclose(T[:3, 3], _to_np(pose.translation), atol=1e-12)

    @randomized
    def test_from_tf_mat_roundtrip_4x4_and_3x4(self, seed):
        pose = self.random_pose(_rng(seed))
        T = _to_np(pose.transform_matrix)
        assert_pose_close(self.Pose.from_tf_mat(T), pose)
        assert_pose_close(self.Pose.from_tf_mat(T[:3]), pose)

    @randomized
    def test_from_rot_mat_roundtrip(self, seed):
        pose = self.random_pose(_rng(seed))
        R, t = _to_np(pose.rotation_matrix), _to_np(pose.translation)
        assert_pose_close(self.Pose.from_rot_mat(R, t), pose)

    @randomized
    def test_from_euler_single_axis_matches_rot_vec(self, seed):
        rng = _rng(seed)
        axis = str(rng.choice(["x", "y", "z"]))
        angle = float(rng.uniform(-np.pi, np.pi))
        e = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
        expect = self.Pose.from_rot_vec(np.array(e, float) * angle)
        assert_pose_close(self.Pose.from_euler(angle, axis), expect)
        assert_pose_close(self.Pose.from_euler(np.degrees(angle), axis, degrees=True), expect)

    @randomized
    def test_from_tf_mat_rejects_bad_shape(self, seed):
        rng = _rng(seed)
        shapes = [(3, 3), (4, 3), (2, 2), (4, 4, 4), (3, 5), (4,)]
        bad = shapes[int(rng.integers(len(shapes)))]
        with pytest.raises(ValueError):
            self.Pose.from_tf_mat(np.zeros(bad))

    # ----- core group operations -------------------------------------------- #

    @randomized
    def test_inverse_undoes_apply(self, seed):
        rng = _rng(seed)
        pose = self.random_pose(rng)
        pts = random_points(rng, (int(rng.integers(1, 8)), 3))
        np.testing.assert_allclose(
            _to_np(pose.inverse().apply(pose.apply(pts))), pts, atol=ATOL
        )

    @randomized
    def test_compose_with_inverse_is_identity(self, seed):
        pose = self.random_pose(_rng(seed))
        assert_pose_close(pose.compose(pose.inverse()), self.Pose.identity())
        assert_pose_close(pose.inverse().compose(pose), self.Pose.identity())

    @randomized
    def test_compose_matches_sequential_apply(self, seed):
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        pts = random_points(rng, (int(rng.integers(1, 8)), 3))
        np.testing.assert_allclose(
            _to_np(a.compose(b).apply(pts)), _to_np(a.apply(b.apply(pts))), atol=ATOL
        )

    @randomized
    def test_compose_matches_matrix_product(self, seed):
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        np.testing.assert_allclose(
            _to_np(a.compose(b).transform_matrix),
            _to_np(a.transform_matrix) @ _to_np(b.transform_matrix),
            atol=ATOL,
        )

    @randomized
    def test_compose_is_associative(self, seed):
        rng = _rng(seed)
        a, b, c = self.random_pose(rng), self.random_pose(rng), self.random_pose(rng)
        assert_pose_close(a.compose(b).compose(c), a.compose(b.compose(c)))

    # ----- Lie-group surface ------------------------------------------------ #

    def test_exp_zero_is_identity(self):
        # exact zero twist exercises the small-angle Taylor branch.
        assert_pose_close(self.Pose.exp(np.zeros(6)), self.Pose.identity())

    @randomized
    def test_exp_pure_translation(self, seed):
        rho = random_trans(_rng(seed))
        pose = self.Pose.exp(np.concatenate([rho, np.zeros(3)]))
        np.testing.assert_allclose(_to_np(pose.quaternion), [0, 0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(_to_np(pose.translation), rho, atol=ATOL)

    @randomized
    def test_exp_pure_rotation(self, seed):
        phi = random_rot_vec(_rng(seed), np.pi - 1e-6)
        pose = self.Pose.exp(np.concatenate([np.zeros(3), phi]))
        assert_pose_close(pose, self.Pose.from_rot_vec(phi))
        np.testing.assert_allclose(_to_np(pose.translation), [0, 0, 0], atol=ATOL)

    @randomized
    def test_exp_log_roundtrip(self, seed):
        xi = random_tangent(_rng(seed))
        np.testing.assert_allclose(_to_np(self.Pose.exp(xi).log()), xi, atol=ATOL)

    @randomized
    def test_exp_log_roundtrip_tiny_angle(self, seed):
        # rotation below _EPS -> Taylor branch of V / V^-1.
        rng = _rng(seed)
        xi = np.concatenate([random_trans(rng, 1.0), random_rot_vec(rng, 1e-10)])
        np.testing.assert_allclose(_to_np(self.Pose.exp(xi).log()), xi, atol=1e-10)

    @randomized
    def test_log_exp_roundtrip(self, seed):
        pose = self.random_pose(_rng(seed))
        assert_pose_close(self.Pose.exp(pose.log()), pose)

    @randomized
    def test_adjoint_defining_identity(self, seed):
        # T ∘ exp(ξ) == exp(Ad_T ξ) ∘ T
        rng = _rng(seed)
        T, xi = self.random_pose(rng), random_tangent(rng)
        ad_xi = _to_np(T.adjoint()) @ xi
        assert_pose_close(T.compose(self.Pose.exp(xi)), self.Pose.exp(ad_xi).compose(T))

    # ----- derived surface & operators -------------------------------------- #

    @randomized
    def test_relative_satisfies_identity(self, seed):
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        assert_pose_close(a.relative(b).compose(b), a)

    @randomized
    def test_operators_alias_methods(self, seed):
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        assert_pose_close(a * b, a.compose(b))
        assert_pose_close(a @ b, a.compose(b))
        assert_pose_close(~a, a.inverse())
        pts = random_points(rng, (int(rng.integers(1, 8)), 3))
        np.testing.assert_allclose(_to_np(a(pts)), _to_np(a.apply(pts)), atol=1e-12)

    @randomized
    def test_boxminus_boxplus_roundtrip(self, seed):
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        xi = a.boxminus(b)  # by definition: b.boxplus(xi) == a
        assert_pose_close(b.boxplus(xi), a)
        np.testing.assert_allclose(_to_np(a - b), _to_np(xi), atol=1e-12)  # __sub__

    @randomized
    def test_interpolate_hits_endpoints(self, seed):
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        assert_pose_close(a.interpolate(b, 0.0), a)
        assert_pose_close(a.interpolate(b, 1.0), b)

    @randomized
    def test_interpolate_midpoint_is_symmetric(self, seed):
        # geodesic midpoint is the same travelled either direction.
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        assert_pose_close(a.interpolate(b, 0.5), b.interpolate(a, 0.5))

    @randomized
    def test_interpolate_matches_screw_groundtruth(self, seed):
        # Cross-check interpolate() against an independent SE(3) geodesic groundtruth:
        # rotation = quaternion slerp (scipy Slerp); translation = screw path via the
        # SO(3) left Jacobian (NOT a straight-line lerp — they agree only at t∈{0,1}).
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, float(rng.uniform(0.0, 1.0))]:
            got = a.interpolate(b, t)
            exp_quat, exp_trans = interpolate_groundtruth(a, b, t)

            got_quat = _to_np(got.quaternion)
            if np.dot(got_quat, exp_quat) < 0:  # quaternion double cover (q ≡ -q)
                exp_quat = -exp_quat
            np.testing.assert_allclose(got_quat, exp_quat, atol=ATOL,
                                       err_msg=f"rotation mismatch at t={t}")
            np.testing.assert_allclose(_to_np(got.translation), exp_trans, atol=ATOL,
                                       err_msg=f"translation mismatch at t={t}")

    @randomized
    def test_interpolate_rotation_is_slerp(self, seed):
        # Isolate the rotation channel: it is exactly the shortest-arc quaternion
        # slerp, independent of the translation.
        rng = _rng(seed)
        a, b = self.random_pose(rng), self.random_pose(rng)
        for t in [0.0, 0.3, 0.5, 0.8, 1.0]:
            exp_quat, _ = interpolate_groundtruth(a, b, t)
            got_quat = _to_np(a.interpolate(b, t).quaternion)
            if np.dot(got_quat, exp_quat) < 0:  # quaternion double cover (q ≡ -q)
                exp_quat = -exp_quat
            np.testing.assert_allclose(got_quat, exp_quat, atol=ATOL,
                                       err_msg=f"rotation not slerp at t={t}")

    @randomized
    def test_interpolate_translation_is_lerp_iff_no_relative_rotation(self, seed):
        # With no relative rotation the left Jacobian V -> I, so the screw path
        # collapses to the straight-line lerp; with rotation present it must NOT.
        rng = _rng(seed)
        q = random_quat(rng)
        same_rot_a = self.Pose.from_quat(q, random_trans(rng))
        same_rot_b = self.Pose.from_quat(q, random_trans(rng))
        ta, tb = _to_np(same_rot_a.translation), _to_np(same_rot_b.translation)
        for t in [0.0, 0.3, 0.5, 0.8, 1.0]:
            np.testing.assert_allclose(
                _to_np(same_rot_a.interpolate(same_rot_b, t).translation),
                (1.0 - t) * ta + t * tb,
                atol=ATOL,
                err_msg=f"translation not lerp (no relative rotation) at t={t}",
            )


    @randomized
    def test_normalize_yields_unit_and_same_pose(self, seed):
        pose = self.random_pose(_rng(seed))
        n = pose.normalize()
        np.testing.assert_allclose(np.linalg.norm(_to_np(n.quaternion)), 1.0, atol=1e-12)
        assert_pose_close(n, pose)

    # ----- value-type guarantees -------------------------------------------- #

    @randomized
    def test_accessors_return_copies(self, seed):
        pose = self.random_pose(_rng(seed))
        q = pose.quaternion
        snap = _to_np(q).copy()
        q[...] = 999.0
        np.testing.assert_array_equal(_to_np(pose.quaternion), snap)

    @randomized
    def test_repr_contains_classname(self, seed):
        assert self.Pose.__name__ in repr(self.random_pose(_rng(seed)))


# ----------------------------------------------------------------------------- #
# NumPy backend
# ----------------------------------------------------------------------------- #

class TestNumpySE3Pose(SE3PoseContract):
    Pose = NumpySE3Pose

    @randomized
    def test_outputs_are_numpy_arrays(self, seed):
        pose = self.random_pose(_rng(seed))
        for arr in (pose.quaternion, pose.apply(np.zeros(3)), pose.log(),
                    pose.rotation_matrix, pose.transform_matrix, pose.adjoint()):
            assert isinstance(arr, np.ndarray)


# ----------------------------------------------------------------------------- #
# torch backend — adds tensor type, dtype/device, autograd, Euler-vs-scipy
# ----------------------------------------------------------------------------- #

class TestTorchSE3Pose(SE3PoseContract):
    Pose = TorchSE3Pose

    # ----- tensor type ------------------------------------------------------ #

    @randomized
    def test_outputs_are_tensors(self, seed):
        pose = self.random_pose(_rng(seed))
        for t in (pose.quaternion, pose.apply(np.zeros(3)), pose.log(),
                  pose.rotation_matrix, pose.transform_matrix, pose.adjoint()):
            assert isinstance(t, torch.Tensor)

    # ----- dtype / device --------------------------------------------------- #

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_is_preserved(self, dtype):
        q = torch.randn(4, dtype=dtype)
        t = torch.randn(3, dtype=dtype)
        pose = TorchSE3Pose.from_quat(q, t)
        assert pose.quaternion.dtype == dtype
        assert pose.rotation_matrix.dtype == dtype
        assert pose.compose(pose).translation.dtype == dtype
        assert pose.log().dtype == dtype
        assert pose.apply(torch.randn(5, 3, dtype=dtype)).dtype == dtype

    def test_arraylike_input_defaults_to_float64(self):
        assert TorchSE3Pose.from_quat([0.0, 0.0, 0.0, 1.0]).quaternion.dtype == torch.float64
        assert TorchSE3Pose.identity().quaternion.dtype == torch.float64
        assert TorchSE3Pose.exp(np.zeros(6)).translation.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @randomized
    def test_cuda_device_is_preserved(self, seed):
        rng = _rng(seed)
        q = torch.tensor(random_quat(rng), device="cuda")
        t = torch.tensor(random_trans(rng), device="cuda")
        pose = TorchSE3Pose.from_quat(q, t)
        assert pose.quaternion.device.type == "cuda"
        assert pose.compose(pose.inverse()).translation.device.type == "cuda"
        assert pose.apply(torch.randn(5, 3, device="cuda", dtype=pose.quaternion.dtype)).device.type == "cuda"
        assert pose.log().device.type == "cuda"

    # ----- autograd flows through the ops ----------------------------------- #

    @randomized
    def test_grad_flows_through_apply(self, seed):
        rng = _rng(seed)
        q = torch.tensor(random_quat(rng), requires_grad=True)
        t = torch.tensor(random_trans(rng), requires_grad=True)
        pose = TorchSE3Pose.from_quat(q, t)
        loss = pose.apply(torch.tensor(random_points(rng, (5, 3)))).sum()
        loss.backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert t.grad is not None and torch.isfinite(t.grad).all()

    @randomized
    def test_grad_flows_through_compose_and_log(self, seed):
        rng = _rng(seed)
        q = torch.tensor(random_quat(rng), requires_grad=True)
        t = torch.tensor(random_trans(rng), requires_grad=True)
        a = TorchSE3Pose.from_quat(q, t)
        b = self.random_pose(rng)
        loss = a.compose(b).log().pow(2).sum()
        loss.backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        assert t.grad is not None and torch.isfinite(t.grad).all()

    @randomized
    def test_grad_flows_through_exp(self, seed):
        xi = torch.tensor(random_tangent(_rng(seed)), requires_grad=True)
        pose = TorchSE3Pose.exp(xi)
        (pose.translation.sum() + pose.rotation_matrix.sum()).backward()
        assert xi.grad is not None and torch.isfinite(xi.grad).all()

    def test_requires_grad_propagates(self):
        q = torch.tensor([0.1, 0.2, 0.3, 1.0], requires_grad=True)
        assert TorchSE3Pose.from_quat(q).quaternion.requires_grad

    # ----- Euler matches the scipy convention (the NumPy backend) ----------- #

    @randomized
    def test_from_euler_matches_numpy_backend(self, seed):
        rng = _rng(seed)
        seq = str(rng.choice(["xyz", "zyx", "XYZ", "ZYX", "xyx", "ZXZ"]))
        angles = rng.uniform(-np.pi, np.pi, size=len(seq))
        np.testing.assert_allclose(
            _to_np(TorchSE3Pose.from_euler(angles, seq).rotation_matrix),
            NumpySE3Pose.from_euler(angles, seq).rotation_matrix,
            atol=1e-6,
        )
