from abc import ABC, abstractmethod
from typing import Any, Union, Optional, List

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


class BaseSE3Trajectory(ABC):
    def __init__(
        self,
        timestamps: List[float],
        poses: List[Any],
        source: Optional[Union[str, int]],
        target: Optional[Union[str, int]],
        normalize: bool = True,
    ):
        self._tstamps = timestamps  # dimension = [N], format = [t]
        self._poses = poses  # dimension = [N, 6], format = [qx, qy, qz, qw, tx, ty, tz]
        self._src = source  # source frame
        # target frame, SE(3) define the coordinate transform from source to target
        #   e.g. source="vehicle", target="world"
        self._dst = target
        self._normalize = normalize

    @classmethod
    @abstractmethod
    def identity(
        cls,
        num: int = 1,
        timestamps: Optional[List[float]] = None,
        source: Optional[Union[str, int]] = None,
        target: Optional[Union[str, int]] = None,
    ) -> "BaseSE3Trajectory":
        """Identity trajectory of ``num`` frames (every pose is the identity SE(3)).

        Emits ``num`` rows of ``[0, 0, 0, 1, 0, 0, 0]`` (identity quaternion xyzw +
        zero translation) as plain Python data; the concrete backend (``cls``)
        materialises them in its own array type.
        """

    def __len__(self) -> int:
        """Number of frames N."""
        return len(self._poses)

    def __getitem__(self, index: Any) -> "BaseSE3Trajectory":
        """Select frames by ``index`` (int / slice / list / array / bool mask),
        returning a new sub-trajectory of the same type.

        A scalar index selects a single frame as a length-1 trajectory (the batch
        dimension is restored); timestamps follow the same selection.
        """
        if isinstance(index, int):
            index = [index]

        return type(self)(
            self._tstamps[index],
            self._poses[index],
            self._src,
            self._dst,
            normalize=self._normalize,
        )

    @staticmethod
    def log(se3_group: Any) -> Any:
        """Log mapping: SE(3) group [N, 4, 4] -> se(3) algebra [N, 6] (batched)."""

    @staticmethod
    def exp(se3_algebra: Any) -> Any:
        """Exp mapping: se(3) algebra [N, 6] -> SE(3) group [N, 4, 4] (batched)."""

    @staticmethod
    def left_jacobian(se3_group: Any):
        """Left jacobian J [N, 6, 6] for SE(3) group [N, 4, 4] / algebra [N, 6]."""

    @abstractmethod
    def transform_matrix(self) -> Any:
        """As transform matrix, dimension = [N, 4, 4]."""

    @abstractmethod
    def rotation_matrix(self) -> Any:
        """As rotation matrix, dimension = [N, 3, 3]."""

    @abstractmethod
    def rotation_vector(self) -> Any:
        """As rotation vector, dimension = [N, 3]."""

    @abstractmethod
    def quaternion(self, scalar_last: bool = True) -> Any:
        """As quaternion, dimension = [N, 4], format = [qx, qy, qz, qw]."""

    @abstractmethod
    def translation(self) -> Any:
        """Return translation component."""

    @abstractmethod
    def euler_angles(self, convention: str = "ZYX", degress: bool = True) -> Any:
        """As Euler Angles, dimension = [N, 3], e.g. ZYX -> [yaw, pitch, roll]."""

    @abstractmethod
    def se3_algebra(self) -> Any:
        """As se(3) algebra, dimension = [N, 6]."""

    @abstractmethod
    def se3_group(self) -> Any:
        """As SE(3) group, dimension = [N, 4, 4]."""

    @abstractmethod
    def norm(self) -> Any:
        """Compute SE(3) norm, i.e. norm of se(3) algebra."""

    @abstractmethod
    def adjoint(self) -> Any:
        """As SE(3) adjoint Ad_T, dimension = [N, 6, 6] (for (rho, phi) twist order)."""

    @abstractmethod
    def apply(self, points: Any) -> Any:
        """Apply SE(3) transform to 3D points."""

    @abstractmethod
    def align(self, other: "BaseSE3Trajectory") -> "BaseSE3Trajectory":
        """Align 2 trajectories with Umeyama method."""

    @abstractmethod
    def relative(self, other: "BaseSE3Trajectory") -> "BaseSE3Trajectory":
        """Compute relative SE(3) pose between 2 trajectories."""

    @abstractmethod
    def interpolate(self, timestamps: List[float]):
        """Interpolate by specific timestamps."""

    @abstractmethod
    def compose(self, other: "BaseSE3Trajectory") -> "BaseSE3Trajectory":
        """Element-wise compose SE(3) poses."""

    @abstractmethod
    def inverse(self) -> "BaseSE3Trajectory":
        """Inverse the SE(3) trajectory."""

    @abstractmethod
    def boxplus(self, twist: Any) -> "BaseSE3Trajectory":
        """Right-plus manifold update ``self ∘ exp(twist)``; twist dimension = [N, 6]."""

    @abstractmethod
    def boxminus(self, other: "BaseSE3Trajectory") -> Any:
        """Right-minus: the [N, 6] twist with ``other.boxplus(result) == self``,
        i.e. ``log(other^{-1} ∘ self)``."""

    @abstractmethod
    def consecutive_twist(self) -> Any:
        """Per-step body twist between consecutive frames, dimension = [N-1, 6]:
        ``log(P_i^{-1} P_{i+1})`` (the arc-length 'motion' measure)."""

    def fit(self) -> Any:
        """Fit SE(3) trajectory to a smooth spline. Future API."""
        raise NotImplementedError


class NumpySE3Trajectory(BaseSE3Trajectory):
    """NumPy backend for :class:`BaseSE3Trajectory` (CPU, float64, no autograd).

    The ``N`` poses are held in ``self._poses`` as a single ``(N, 7)`` array of
    ``[qx, qy, qz, qw, tx, ty, tz]`` rows (unit quaternion scalar-last +
    translation) — the canonical state declared by the base. SO(3) conversions are
    delegated to ``scipy.spatial.transform.Rotation`` (also scalar-last); the
    SE(3) exp/log translation coupling uses the closed-form SO(3) left Jacobian.

    Conventions: active transform ``x' = R @ x + t`` mapping ``source -> target``;
    the ``se(3)`` twist is ordered ``xi = (rho, phi)`` with the translation part
    ``rho`` first and the ``so(3)`` part ``phi`` second; ``se3_algebra`` /
    ``rotation_vector`` angles are radians. Every accessor keeps the batch axis
    (``N``), so a single frame is ``[1, ...]``, never squeezed.

    Implements exactly the :class:`BaseSE3Trajectory` API (no extra members).
    """

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        timestamps: Optional[Any],
        poses: Any,
        source: Optional[Union[str, int]] = None,
        target: Optional[Union[str, int]] = None,
        *,
        normalize: bool = True,
    ):
        """Store ``(N, 7)`` ``[qx, qy, qz, qw, tx, ty, tz]`` poses + optional time axis.

        Args:
            timestamps: ``(N,)`` seconds, or ``None`` (index-parameterised).
            poses: ``(N, 7)`` array-like of ``[quat xyzw, translation]`` rows.
            source / target: frame labels; SE(3) maps ``source -> target``.
            normalize: re-project each quaternion to unit length.
        """
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(
                f"poses must be (N, 7) [qx, qy, qz, qw, tx, ty, tz], got {poses.shape}"
            )
        if normalize:
            n = np.linalg.norm(poses[:, :4], axis=1, keepdims=True)
            if np.any(n < 1e-8):
                raise ValueError("a quaternion has ~zero norm; cannot normalize")
            poses = poses.copy()
            poses[:, :4] = poses[:, :4] / n
        if timestamps is not None:
            timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
            if timestamps.shape[0] != poses.shape[0]:
                raise ValueError(
                    f"timestamps length {timestamps.shape[0]} != N={poses.shape[0]}"
                )
        super().__init__(timestamps, poses, source, target, normalize)

    @classmethod
    def identity(
        cls,
        num: int = 1,
        timestamps: Optional[List[float]] = None,
        source: Optional[Union[str, int]] = None,
        target: Optional[Union[str, int]] = None,
    ) -> "NumpySE3Trajectory":
        poses = np.zeros((num, 7))
        poses[:, 3] = 1.0  # qw = 1 -> identity rotation
        return cls(timestamps, poses, source, target, normalize=False)

    # ------------------------------------------------------------------ #
    # representation converters (batched over N)
    # ------------------------------------------------------------------ #
    def transform_matrix(self) -> np.ndarray:
        tf_mat = np.broadcast_to(np.eye(4), (self._poses.shape[0], 4, 4)).copy()
        tf_mat[:, :3, :3] = Rotation.from_quat(self._poses[:, :4]).as_matrix()
        tf_mat[:, :3, 3] = self._poses[:, 4:7]
        return tf_mat

    def se3_group(self) -> np.ndarray:
        return self.transform_matrix()

    def rotation_matrix(self) -> np.ndarray:
        return Rotation.from_quat(self._poses[:, :4]).as_matrix()

    def rotation_vector(self) -> np.ndarray:
        return Rotation.from_quat(self._poses[:, :4]).as_rotvec()

    def quaternion(self, scalar_last: bool = True) -> np.ndarray:
        quat = self._poses[:, :4].copy()
        return quat if scalar_last else quat[:, [3, 0, 1, 2]]

    def translation(self) -> np.ndarray:
        return self._poses[:, 4:7].copy()

    def euler_angles(self, convention: str = "ZYX", degress: bool = True) -> np.ndarray:
        # NB: the `degress` keyword mirrors the (misspelled) base signature.
        return Rotation.from_quat(self._poses[:, :4]).as_euler(convention, degrees=degress)

    def se3_algebra(self) -> np.ndarray:
        return self.log(self.transform_matrix())

    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.se3_algebra(), axis=1)

    def adjoint(self) -> np.ndarray:
        tf_mat = self.transform_matrix()
        rot, trans = tf_mat[:, :3, :3], tf_mat[:, :3, 3]
        skew_t = np.zeros((tf_mat.shape[0], 3, 3))
        skew_t[:, 0, 1], skew_t[:, 0, 2] = -trans[:, 2], trans[:, 1]
        skew_t[:, 1, 0], skew_t[:, 1, 2] = trans[:, 2], -trans[:, 0]
        skew_t[:, 2, 0], skew_t[:, 2, 1] = -trans[:, 1], trans[:, 0]
        ad = np.zeros((tf_mat.shape[0], 6, 6))
        ad[:, :3, :3] = rot
        ad[:, :3, 3:] = skew_t @ rot  # (rho, phi) ordering: rotation couples into rho
        ad[:, 3:, 3:] = rot
        return ad

    # ------------------------------------------------------------------ #
    # SE(3) group operations (element-wise over frames)
    # ------------------------------------------------------------------ #
    def inverse(self) -> "NumpySE3Trajectory":
        r_inv = Rotation.from_quat(self._poses[:, :4]).inv()
        poses = np.concatenate(
            [r_inv.as_quat(), -r_inv.apply(self._poses[:, 4:7])], axis=1
        )
        # a source->target path becomes target->source
        return type(self)(self._tstamps, poses, self._dst, self._src, normalize=False)

    def compose(self, other: "BaseSE3Trajectory") -> "NumpySE3Trajectory":
        if len(other) != len(self):
            raise ValueError(f"compose length mismatch: {len(self)} vs {len(other)}")
        # frame chaining: other(source->mid) then self(mid->target) == self @ other
        tf_mat = self.transform_matrix() @ other.transform_matrix()
        quat = Rotation.from_matrix(tf_mat[:, :3, :3]).as_quat()
        poses = np.concatenate([quat, tf_mat[:, :3, 3]], axis=1)
        return type(self)(
            self._tstamps, poses, getattr(other, "_src", None), self._dst, normalize=False
        )

    def relative(self, other: "BaseSE3Trajectory") -> "NumpySE3Trajectory":
        """Per-frame ``other_i^{-1} @ self_i`` (``self`` in ``other``'s body frame);
        satisfies ``other.compose(self.relative(other)) == self``."""
        if len(other) != len(self):
            raise ValueError(f"relative length mismatch: {len(self)} vs {len(other)}")
        ts_mat = self.transform_matrix()
        to_mat = other.transform_matrix()
        rot_o = to_mat[:, :3, :3]
        # closed-form inv(other) @ self = (Ro^T Rs, Ro^T (ts - to)); avoids np.linalg.inv
        tf_mat = np.broadcast_to(np.eye(4), ts_mat.shape).copy()
        tf_mat[:, :3, :3] = np.einsum("nji,njk->nik", rot_o, ts_mat[:, :3, :3])
        tf_mat[:, :3, 3] = np.einsum(
            "nji,nj->ni", rot_o, ts_mat[:, :3, 3] - to_mat[:, :3, 3]
        )
        quat = Rotation.from_matrix(tf_mat[:, :3, :3]).as_quat()
        poses = np.concatenate([quat, tf_mat[:, :3, 3]], axis=1)
        return type(self)(self._tstamps, poses, self._src, self._dst, normalize=False)

    def boxplus(self, twist: Any) -> "NumpySE3Trajectory":
        """Right-plus: ``self ∘ exp(twist)`` with ``twist`` a ``[N, 6]`` se(3) twist."""
        tf_mat = self.transform_matrix() @ self.exp(twist)
        quat = Rotation.from_matrix(tf_mat[:, :3, :3]).as_quat()
        poses = np.concatenate([quat, tf_mat[:, :3, 3]], axis=1)
        return type(self)(self._tstamps, poses, self._src, self._dst, normalize=False)

    def boxminus(self, other: "BaseSE3Trajectory") -> np.ndarray:
        """Right-minus ``log(other^{-1} ∘ self)`` -> ``[N, 6]``;
        satisfies ``other.boxplus(self.boxminus(other)) == self``."""
        ts_mat = self.transform_matrix()
        to_mat = other.transform_matrix()
        rot_o = to_mat[:, :3, :3]
        # closed-form inv(other) @ self (avoids np.linalg.inv)
        rel = np.broadcast_to(np.eye(4), ts_mat.shape).copy()
        rel[:, :3, :3] = np.einsum("nji,njk->nik", rot_o, ts_mat[:, :3, :3])
        rel[:, :3, 3] = np.einsum("nji,nj->ni", rot_o, ts_mat[:, :3, 3] - to_mat[:, :3, 3])
        return self.log(rel)

    def consecutive_twist(self) -> np.ndarray:
        """Per-step body twist ``log(P_i^{-1} P_{i+1})`` -> ``[N-1, 6]``."""
        tf_mat = self.transform_matrix()
        rot_a = tf_mat[:-1, :3, :3]
        # closed-form inv(P_i) @ P_{i+1} for each consecutive pair (avoids np.linalg.inv)
        rel = np.broadcast_to(np.eye(4), (tf_mat.shape[0] - 1, 4, 4)).copy()
        rel[:, :3, :3] = np.einsum("nji,njk->nik", rot_a, tf_mat[1:, :3, :3])
        rel[:, :3, 3] = np.einsum(
            "nji,nj->ni", rot_a, tf_mat[1:, :3, 3] - tf_mat[:-1, :3, 3]
        )
        return self.log(rel)

    def apply(self, points: Any) -> np.ndarray:
        """Transform 3D points by the trajectory.

        Shapes: ``(3,)`` or ``(M, 3)`` -> broadcast every frame to the same points,
        returning ``(N, 3)`` / ``(N, M, 3)``; ``(N, 3)`` -> one point per frame
        (paired) -> ``(N, 3)``; ``(N, M, 3)`` -> paired clouds -> ``(N, M, 3)``.
        """
        points = np.asarray(points, dtype=np.float64)
        R = self.rotation_matrix()
        t = self._poses[:, 4:7]
        num = self._poses.shape[0]
        if points.shape == (3,):  # single point, broadcast to all frames
            return R @ points + t
        if points.ndim == 2 and points.shape == (
            num,
            3,
        ):  # one point per frame (paired)
            return np.einsum("nij,nj->ni", R, points) + t
        if points.ndim == 2 and points.shape[1] == 3:  # (M, 3) cloud, broadcast
            return np.einsum("nij,mj->nmi", R, points) + t[:, None, :]
        if (
            points.ndim == 3 and points.shape[0] == num and points.shape[2] == 3
        ):  # (N, M, 3) paired
            return np.einsum("nij,nmj->nmi", R, points) + t[:, None, :]
        raise ValueError(
            f"unsupported points shape {points.shape}; expected (3,), (M, 3), (N, 3) or (N, M, 3)"
        )

    # ------------------------------------------------------------------ #
    # temporal & registration ops
    # ------------------------------------------------------------------ #
    def interpolate(self, timestamps: Any) -> "NumpySE3Trajectory":
        """Resample at ``timestamps`` along the SE(3) geodesic (constant-twist screw
        motion) between bracketing frames. Out-of-range queries raise (no
        extrapolation). Fully vectorised over the query times."""
        if self._tstamps is None:
            raise ValueError("trajectory has no timestamps; cannot interpolate")
        if len(self) < 2:
            raise ValueError("interpolate needs at least 2 frames")
        t_src = np.asarray(self._tstamps, dtype=np.float64)
        t_query = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if t_query.min() < t_src[0] - 1e-8 or t_query.max() > t_src[-1] + 1e-8:
            raise ValueError("query timestamps out of range; no extrapolation")
        idx = np.clip(
            np.searchsorted(t_src, t_query, side="right") - 1, 0, len(self) - 2
        )
        t0, t1 = t_src[idx], t_src[idx + 1]
        alpha = np.where(t1 > t0, (t_query - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
        tf_mat = self.transform_matrix()
        rel = np.linalg.inv(tf_mat[idx]) @ tf_mat[idx + 1]  # (M, 4, 4)
        delta = self.exp(alpha[:, None] * self.log(rel))  # (M, 4, 4)
        out = tf_mat[idx] @ delta  # (M, 4, 4)
        quat = Rotation.from_matrix(out[:, :3, :3]).as_quat()
        poses = np.concatenate([quat, out[:, :3, 3]], axis=1)
        return type(self)(t_query, poses, self._src, self._dst, normalize=False)

    def align(self, other: "BaseSE3Trajectory") -> "NumpySE3Trajectory":
        """Umeyama-align ``self`` onto ``other`` over the matched per-frame positions,
        returning the aligned copy of ``self`` (rigid SE(3); no scale).

        Delegates to ``evo``'s ``umeyama_alignment`` (the same routine the camera-pose
        ATE metric uses) so the two cannot silently diverge."""
        from evo.core.geometry import umeyama_alignment  # lazy: evo is heavy

        if len(other) != len(self):
            raise ValueError(f"align length mismatch: {len(self)} vs {len(other)}")
        src = self.translation()  # (N, 3)
        dst = np.asarray(other.transform_matrix()[:, :3, 3], dtype=np.float64)
        # evo wants (3, N) points and returns (rotation, translation, scale).
        R, t, _ = umeyama_alignment(src.T, dst.T, with_scale=False)
        R_new = R @ self.rotation_matrix()  # (N, 3, 3)
        t_new = (self._poses[:, 4:7] @ R.T) + t  # (N, 3)
        poses = np.concatenate([Rotation.from_matrix(R_new).as_quat(), t_new], axis=1)
        return type(self)(self._tstamps, poses, self._src, self._dst, normalize=False)

    # ------------------------------------------------------------------ #
    # Lie-group utilities (batched; a single element keeps the batch axis -> [1, ...])
    # ------------------------------------------------------------------ #
    @staticmethod
    def log(se3_group: Any) -> np.ndarray:
        """SE(3) log: ``[N, 4, 4]`` group -> ``[N, 6]`` twist ``(rho, phi)``
        (translation first). A single ``[4, 4]`` is promoted to ``[1, 4, 4]``; the
        batch axis is always kept (output ``[N, 6]``, never ``[6]``)."""
        tf = np.asarray(se3_group, dtype=np.float64)
        if tf.ndim == 2:
            tf = tf[None]
        phi = Rotation.from_matrix(tf[:, :3, :3]).as_rotvec()  # (N, 3)
        theta = np.linalg.norm(phi, axis=1)
        skew = np.zeros((phi.shape[0], 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -phi[:, 2], phi[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = phi[:, 2], -phi[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -phi[:, 1], phi[:, 0]
        eye = np.broadcast_to(np.eye(3), skew.shape).copy()
        small = theta < 1e-8
        c = np.full_like(theta, 1.0 / 12.0)  # inverse SO(3) left-Jacobian coeff
        th = theta[~small]
        c[~small] = 1.0 / th**2 - (1.0 + np.cos(th)) / (2.0 * th * np.sin(th))
        v_inv = eye - 0.5 * skew + c[:, None, None] * (skew @ skew)
        rho = np.einsum("nij,nj->ni", v_inv, tf[:, :3, 3])
        return np.concatenate([rho, phi], axis=1)

    @staticmethod
    def exp(se3_algebra: Any) -> np.ndarray:
        """SE(3) exp: ``[N, 6]`` twist ``(rho, phi)`` (translation first) -> ``[N, 4, 4]``
        group. A single ``[6]`` is promoted to ``[1, 6]``; the batch axis is always
        kept (output ``[N, 4, 4]``, never ``[4, 4]``)."""
        xi = np.asarray(se3_algebra, dtype=np.float64)
        if xi.ndim == 1:
            xi = xi[None]
        rho, phi = xi[:, :3], xi[:, 3:]
        theta = np.linalg.norm(phi, axis=1)
        skew = np.zeros((xi.shape[0], 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -phi[:, 2], phi[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = phi[:, 2], -phi[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -phi[:, 1], phi[:, 0]
        eye = np.broadcast_to(np.eye(3), skew.shape).copy()
        small = theta < 1e-8
        a = np.full_like(theta, 0.5)  # SO(3) left-Jacobian coeffs
        b = np.full_like(theta, 1.0 / 6.0)
        th = theta[~small]
        a[~small] = (1.0 - np.cos(th)) / th**2
        b[~small] = (th - np.sin(th)) / th**3
        v_mat = eye + a[:, None, None] * skew + b[:, None, None] * (skew @ skew)
        tf = np.broadcast_to(np.eye(4), (xi.shape[0], 4, 4)).copy()
        tf[:, :3, :3] = Rotation.from_rotvec(phi).as_matrix()
        tf[:, :3, 3] = np.einsum("nij,nj->ni", v_mat, rho)
        return tf

    @staticmethod
    def left_jacobian(se3_group: Any) -> np.ndarray:
        """SE(3) left Jacobian ``J_l`` for the ``(rho, phi)`` twist ordering.

        Accepts a group element ``[N, 4, 4]`` (logged first) or a twist ``[N, 6]``; a
        single ``[4, 4]`` / ``[6]`` is promoted, so the output is always ``[N, 6, 6]``
        (never ``[6, 6]``). Computed exactly via the Van Loan integral
        ``J_l = ∫_0^1 exp(s·ad) ds`` (top-right block of ``expm([[ad, I], [0, 0]])``),
        valid for all twists with no small-angle case.
        """
        arr = np.asarray(se3_group, dtype=np.float64)
        if arr.shape[-2:] == (4, 4):
            xi = NumpySE3Trajectory.log(arr)  # already batched [N, 6]
        else:
            xi = arr[None] if arr.ndim == 1 else arr
        jac = np.empty((xi.shape[0], 6, 6))
        for k in range(xi.shape[0]):
            rx, ry, rz = xi[k, :3]
            px, py, pz = xi[k, 3:]
            skew_phi = np.array([[0.0, -pz, py], [pz, 0.0, -px], [-py, px, 0.0]])
            skew_rho = np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])
            ad = np.zeros((6, 6))
            ad[:3, :3] = skew_phi
            ad[:3, 3:] = skew_rho
            ad[3:, 3:] = skew_phi
            aug = np.zeros((12, 12))
            aug[:6, :6], aug[:6, 6:] = ad, np.eye(6)
            jac[k] = expm(aug)[:6, 6:]
        return jac
