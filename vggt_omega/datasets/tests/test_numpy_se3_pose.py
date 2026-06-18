"""Randomized behaviour/property tests for :class:`NumpySE3Pose`.

Every property is checked against many random SE(3) poses (parametrized over
seeds), so the group/Lie laws and round-trips are stressed across the whole
manifold rather than a handful of hand-picked poses. Conventions (scalar-last
quaternion, right-handed ``x' = Rx + t``, translation-first twist) are pinned by
randomizing the *inputs* while asserting the convention-defining outcome.
"""
import numpy as np
import pytest

from vggt_omega.datasets.se3_pose import NumpySE3Pose

# Each randomized test runs over this many independent random cases.
N_RANDOM = 50
randomized = pytest.mark.parametrize("seed", range(N_RANDOM))

ATOL = 1e-8  # comfortably above the ~1e-10 worst-case numerical floor


# ----------------------------------------------------------------------------- #
# random generators
# ----------------------------------------------------------------------------- #

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_quat(rng: np.random.Generator) -> np.ndarray:
    """Uniform on SO(3): a normalized 4D Gaussian is uniform on S^3 (xyzw)."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def random_trans(rng: np.random.Generator, scale: float = 5.0) -> np.ndarray:
    return rng.uniform(-scale, scale, size=3)


def random_pose(rng: np.random.Generator) -> NumpySE3Pose:
    return NumpySE3Pose.from_quat(random_quat(rng), random_trans(rng))


def random_rot_vec(rng: np.random.Generator, max_angle: float = np.pi) -> np.ndarray:
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    return axis * rng.uniform(0.0, max_angle)


def random_tangent(rng: np.random.Generator) -> np.ndarray:
    """Twist ``(ρ, φ)`` with ``‖φ‖ < π`` so ``exp`` is injective (``log`` inverts)."""
    return np.concatenate([random_trans(rng), random_rot_vec(rng, np.pi - 1e-6)])


def random_points(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.uniform(-10.0, 10.0, size=shape)


def assert_pose_close(a: NumpySE3Pose, b: NumpySE3Pose, atol: float = ATOL) -> None:
    """Equal poses, accounting for the quaternion double cover (q ≡ -q)."""
    qa = np.asarray(a.quaternion, float)
    qb = np.asarray(b.quaternion, float)
    if np.dot(qa, qb) < 0:
        qb = -qb
    np.testing.assert_allclose(qa, qb, atol=atol)
    np.testing.assert_allclose(a.translation, b.translation, atol=atol)


# ----------------------------------------------------------------------------- #
# construction & accessors
# ----------------------------------------------------------------------------- #

@randomized
def test_identity_is_noop(seed):
    rng = _rng(seed)
    ident = NumpySE3Pose.identity()
    np.testing.assert_allclose(ident.quaternion, [0, 0, 0, 1], atol=1e-12)
    pts = random_points(rng, (int(rng.integers(1, 8)), 3))
    np.testing.assert_allclose(ident.apply(pts), pts, atol=1e-12)


@randomized
def test_from_quat_normalizes(seed):
    rng = _rng(seed)
    q = random_quat(rng)
    scaled = q * rng.uniform(0.1, 10.0)  # off the unit sphere
    pose = NumpySE3Pose.from_quat(scaled)
    np.testing.assert_allclose(np.linalg.norm(pose.quaternion), 1.0, atol=1e-12)
    assert_pose_close(pose, NumpySE3Pose.from_quat(q))  # same rotation


@randomized
def test_from_quat_wxyz_is_reordered(seed):
    rng = _rng(seed)
    q_xyzw = random_quat(rng)
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    assert_pose_close(
        NumpySE3Pose.from_quat(q_wxyz, scalar_last=False),
        NumpySE3Pose.from_quat(q_xyzw),
    )


@randomized
def test_from_quat_default_zero_translation(seed):
    pose = NumpySE3Pose.from_quat(random_quat(_rng(seed)))
    np.testing.assert_allclose(pose.translation, [0, 0, 0], atol=1e-12)


@randomized
def test_apply_is_right_handed_about_z(seed):
    # +θ about z maps e_x -> (cosθ, sinθ, 0): pins right-handedness, random angle.
    theta = _rng(seed).uniform(-np.pi, np.pi)
    pose = NumpySE3Pose.from_rot_vec([0, 0, theta])
    np.testing.assert_allclose(pose.apply([1, 0, 0]), [np.cos(theta), np.sin(theta), 0], atol=1e-9)


@randomized
def test_apply_matches_rotation_matrix_form(seed):
    rng = _rng(seed)
    pose = random_pose(rng)
    pts = random_points(rng, (int(rng.integers(1, 8)), 3))
    np.testing.assert_allclose(
        pose.apply(pts), pts @ pose.rotation_matrix.T + pose.translation, atol=ATOL
    )


@randomized
def test_apply_preserves_point_array_shape(seed):
    rng = _rng(seed)
    pose = random_pose(rng)
    for shape in [(3,), (5, 3), (2, 4, 3)]:
        assert pose.apply(random_points(rng, shape)).shape == shape


@randomized
def test_rotation_matrix_is_special_orthogonal(seed):
    R = random_pose(_rng(seed)).rotation_matrix
    assert R.shape == (3, 3)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


@randomized
def test_transform_matrix_blocks(seed):
    pose = random_pose(_rng(seed))
    T = pose.transform_matrix
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3], pose.rotation_matrix, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], pose.translation, atol=1e-12)


@randomized
def test_from_tf_mat_roundtrip_4x4_and_3x4(seed):
    pose = random_pose(_rng(seed))
    T = pose.transform_matrix
    assert_pose_close(NumpySE3Pose.from_tf_mat(T), pose)
    assert_pose_close(NumpySE3Pose.from_tf_mat(T[:3]), pose)


@randomized
def test_from_rot_mat_roundtrip(seed):
    pose = random_pose(_rng(seed))
    assert_pose_close(
        NumpySE3Pose.from_rot_mat(pose.rotation_matrix, pose.translation), pose
    )


@randomized
def test_from_euler_single_axis_matches_rot_vec(seed):
    rng = _rng(seed)
    axis = str(rng.choice(["x", "y", "z"]))
    angle = rng.uniform(-np.pi, np.pi)
    e = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    expect = NumpySE3Pose.from_rot_vec(np.array(e, float) * angle)
    assert_pose_close(NumpySE3Pose.from_euler(angle, axis), expect)
    assert_pose_close(NumpySE3Pose.from_euler(np.degrees(angle), axis, degrees=True), expect)


@randomized
def test_from_tf_mat_rejects_bad_shape(seed):
    rng = _rng(seed)
    shapes = [(3, 3), (4, 3), (2, 2), (4, 4, 4), (3, 5), (4,)]
    bad = shapes[int(rng.integers(len(shapes)))]
    with pytest.raises(ValueError):
        NumpySE3Pose.from_tf_mat(np.zeros(bad))


# ----------------------------------------------------------------------------- #
# core group operations
# ----------------------------------------------------------------------------- #

@randomized
def test_inverse_undoes_apply(seed):
    rng = _rng(seed)
    pose = random_pose(rng)
    pts = random_points(rng, (int(rng.integers(1, 8)), 3))
    np.testing.assert_allclose(pose.inverse().apply(pose.apply(pts)), pts, atol=ATOL)


@randomized
def test_compose_with_inverse_is_identity(seed):
    pose = random_pose(_rng(seed))
    assert_pose_close(pose.compose(pose.inverse()), NumpySE3Pose.identity())
    assert_pose_close(pose.inverse().compose(pose), NumpySE3Pose.identity())


@randomized
def test_compose_matches_sequential_apply(seed):
    rng = _rng(seed)
    a, b = random_pose(rng), random_pose(rng)
    pts = random_points(rng, (int(rng.integers(1, 8)), 3))
    np.testing.assert_allclose(a.compose(b).apply(pts), a.apply(b.apply(pts)), atol=ATOL)


@randomized
def test_compose_matches_matrix_product(seed):
    rng = _rng(seed)
    a, b = random_pose(rng), random_pose(rng)
    np.testing.assert_allclose(
        a.compose(b).transform_matrix, a.transform_matrix @ b.transform_matrix, atol=ATOL
    )


@randomized
def test_compose_is_associative(seed):
    rng = _rng(seed)
    a, b, c = random_pose(rng), random_pose(rng), random_pose(rng)
    assert_pose_close(a.compose(b).compose(c), a.compose(b.compose(c)))


# ----------------------------------------------------------------------------- #
# Lie-group surface
# ----------------------------------------------------------------------------- #

def test_exp_zero_is_identity():
    # exact zero twist exercises the small-angle Taylor branch.
    assert_pose_close(NumpySE3Pose.exp(np.zeros(6)), NumpySE3Pose.identity())


@randomized
def test_exp_pure_translation(seed):
    rho = random_trans(_rng(seed))
    pose = NumpySE3Pose.exp(np.concatenate([rho, np.zeros(3)]))
    np.testing.assert_allclose(pose.quaternion, [0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(pose.translation, rho, atol=1e-12)


@randomized
def test_exp_pure_rotation(seed):
    phi = random_rot_vec(_rng(seed), np.pi - 1e-6)
    pose = NumpySE3Pose.exp(np.concatenate([np.zeros(3), phi]))
    assert_pose_close(pose, NumpySE3Pose.from_rot_vec(phi))
    np.testing.assert_allclose(pose.translation, [0, 0, 0], atol=1e-12)


@randomized
def test_exp_log_roundtrip(seed):
    xi = random_tangent(_rng(seed))
    np.testing.assert_allclose(NumpySE3Pose.exp(xi).log(), xi, atol=ATOL)


@randomized
def test_exp_log_roundtrip_tiny_angle(seed):
    # rotation below _EPS -> Taylor branch of V / V^-1.
    rng = _rng(seed)
    xi = np.concatenate([random_trans(rng, 1.0), random_rot_vec(rng, 1e-10)])
    np.testing.assert_allclose(NumpySE3Pose.exp(xi).log(), xi, atol=1e-12)


@randomized
def test_log_exp_roundtrip(seed):
    pose = random_pose(_rng(seed))
    assert_pose_close(NumpySE3Pose.exp(pose.log()), pose)


@randomized
def test_adjoint_defining_identity(seed):
    # T ∘ exp(ξ) == exp(Ad_T ξ) ∘ T
    rng = _rng(seed)
    T, xi = random_pose(rng), random_tangent(rng)
    assert_pose_close(
        T.compose(NumpySE3Pose.exp(xi)),
        NumpySE3Pose.exp(T.adjoint() @ xi).compose(T),
    )


# ----------------------------------------------------------------------------- #
# derived surface & operators
# ----------------------------------------------------------------------------- #

@randomized
def test_relative_satisfies_identity(seed):
    rng = _rng(seed)
    a, b = random_pose(rng), random_pose(rng)
    assert_pose_close(a.relative(b).compose(b), a)


@randomized
def test_operators_alias_methods(seed):
    rng = _rng(seed)
    a, b = random_pose(rng), random_pose(rng)
    assert_pose_close(a * b, a.compose(b))
    assert_pose_close(a @ b, a.compose(b))
    assert_pose_close(~a, a.inverse())
    pts = random_points(rng, (int(rng.integers(1, 8)), 3))
    np.testing.assert_allclose(a(pts), a.apply(pts), atol=1e-12)


@randomized
def test_boxminus_boxplus_roundtrip(seed):
    rng = _rng(seed)
    a, b = random_pose(rng), random_pose(rng)
    xi = a.boxminus(b)  # by definition: b.boxplus(xi) == a
    assert_pose_close(b.boxplus(xi), a)
    np.testing.assert_allclose(a - b, xi, atol=1e-12)  # __sub__ == boxminus


@randomized
def test_interpolate_hits_endpoints(seed):
    rng = _rng(seed)
    a, b = random_pose(rng), random_pose(rng)
    assert_pose_close(a.interpolate(b, 0.0), a)
    assert_pose_close(a.interpolate(b, 1.0), b)


@randomized
def test_interpolate_midpoint_is_symmetric(seed):
    # geodesic midpoint is the same travelled either direction.
    rng = _rng(seed)
    a, b = random_pose(rng), random_pose(rng)
    assert_pose_close(a.interpolate(b, 0.5), b.interpolate(a, 0.5))


@randomized
def test_normalize_yields_unit_and_same_pose(seed):
    pose = random_pose(_rng(seed))
    n = pose.normalize()
    np.testing.assert_allclose(np.linalg.norm(n.quaternion), 1.0, atol=1e-12)
    assert_pose_close(n, pose)


# ----------------------------------------------------------------------------- #
# value-type & backend guarantees
# ----------------------------------------------------------------------------- #

@randomized
def test_outputs_are_numpy_arrays(seed):
    pose = random_pose(_rng(seed))
    for arr in (pose.quaternion, pose.apply(np.zeros(3)), pose.log(),
                pose.rotation_matrix, pose.transform_matrix, pose.adjoint()):
        assert isinstance(arr, np.ndarray)


@randomized
def test_accessors_return_copies(seed):
    pose = random_pose(_rng(seed))
    q = pose.quaternion
    snap = q.copy()
    q[:] = 999.0
    np.testing.assert_array_equal(pose.quaternion, snap)
    t = pose.translation
    tsnap = t.copy()
    t[:] = 999.0
    np.testing.assert_array_equal(pose.translation, tsnap)


@randomized
def test_repr_contains_classname(seed):
    assert "NumpySE3Pose" in repr(random_pose(_rng(seed)))
