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

    @staticmethod
    def log(se3_group: Any) -> Any:
        """Log mapping from SE(3) group to se(3) algebra."""

    @staticmethod
    def exp(se3_algebra: Any) -> Any:
        """Exponential mapping from se(3) algebra to SE(3) group."""

    @staticmethod
    def left_jacobian(se3_group: Any):
        """Compute left jacobian J for SE(3) group."""

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
    ``rotation_vector`` angles are radians.
    """

    _EPS = 1e-8

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
            timestamps: ``(N,)`` seconds, or ``None`` (index-parameterised; disables
                :meth:`interpolate`).
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
            if np.any(n < self._EPS):
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
    def from_tf_mat(
        cls, tf_mat: Any, timestamps: Optional[Any] = None, source=None, target=None
    ) -> "NumpySE3Trajectory":
        """Build from ``(N, 4, 4)`` or ``(N, 3, 4)`` homogeneous transforms."""
        tf_mat = np.asarray(tf_mat, dtype=np.float64)
        if tf_mat.ndim != 3 or tf_mat.shape[1:] not in ((4, 4), (3, 4)):
            raise ValueError(f"tf_mat must be (N, 4, 4) or (N, 3, 4), got {tf_mat.shape}")
        quat = Rotation.from_matrix(tf_mat[:, :3, :3]).as_quat()  # xyzw, (N, 4)
        poses = np.concatenate([quat, tf_mat[:, :3, 3]], axis=1)
        return cls(timestamps, poses, source, target, normalize=False)

    @classmethod
    def from_quat(
        cls,
        quat: Any,
        trans: Optional[Any] = None,
        timestamps: Optional[Any] = None,
        source=None,
        target=None,
        *,
        scalar_last: bool = True,
        normalize: bool = True,
    ) -> "NumpySE3Trajectory":
        """Build from ``(N, 4)`` quaternions (+ optional ``(N, 3)`` translations)."""
        quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
        if not scalar_last:  # wxyz -> xyzw
            quat = quat[:, [1, 2, 3, 0]]
        trans = (
            np.zeros((quat.shape[0], 3))
            if trans is None
            else np.asarray(trans, dtype=np.float64).reshape(-1, 3)
        )
        return cls(
            timestamps,
            np.concatenate([quat, trans], axis=1),
            source,
            target,
            normalize=normalize,
        )

    # ------------------------------------------------------------------ #
    # internal state views & dunders
    # ------------------------------------------------------------------ #
    @property
    def _quat(self) -> np.ndarray:
        return self._poses[:, :4]

    @property
    def _trans(self) -> np.ndarray:
        return self._poses[:, 4:7]

    def _rotation(self) -> Rotation:
        return Rotation.from_quat(self._quat)

    def __len__(self) -> int:
        return self._poses.shape[0]

    def __repr__(self) -> str:
        return f"NumpySE3Trajectory(N={len(self)}, source={self._src!r}, target={self._dst!r})"

    # ------------------------------------------------------------------ #
    # representation converters (batched over N)
    # ------------------------------------------------------------------ #
    def transform_matrix(self) -> np.ndarray:
        tf_mat = np.broadcast_to(np.eye(4), (len(self), 4, 4)).copy()
        tf_mat[:, :3, :3] = self._rotation().as_matrix()
        tf_mat[:, :3, 3] = self._trans
        return tf_mat

    def se3_group(self) -> np.ndarray:
        return self.transform_matrix()

    def rotation_matrix(self) -> np.ndarray:
        return self._rotation().as_matrix()

    def rotation_vector(self) -> np.ndarray:
        return self._rotation().as_rotvec()

    def quaternion(self, scalar_last: bool = True) -> np.ndarray:
        quat = self._quat.copy()
        return quat if scalar_last else quat[:, [3, 0, 1, 2]]

    def translation(self) -> np.ndarray:
        return self._trans.copy()

    def euler_angles(self, convention: str = "ZYX", degress: bool = True) -> np.ndarray:
        # NB: the `degress` keyword mirrors the (misspelled) base signature.
        return self._rotation().as_euler(convention, degrees=degress)

    def se3_algebra(self) -> np.ndarray:
        return self.log(self.transform_matrix())

    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.se3_algebra(), axis=1)

    # ------------------------------------------------------------------ #
    # SE(3) group operations (element-wise over frames)
    # ------------------------------------------------------------------ #
    def inverse(self) -> "NumpySE3Trajectory":
        r_inv = self._rotation().inv()
        poses = np.concatenate([r_inv.as_quat(), -r_inv.apply(self._trans)], axis=1)
        # a source->target path becomes target->source
        return type(self)(self._tstamps, poses, self._dst, self._src, normalize=False)

    def compose(self, other: "BaseSE3Trajectory") -> "NumpySE3Trajectory":
        if len(other) != len(self):
            raise ValueError(f"compose length mismatch: {len(self)} vs {len(other)}")
        # frame chaining: other(source->mid) then self(mid->target) == self @ other
        tf_mat = self.transform_matrix() @ other.transform_matrix()
        return type(self).from_tf_mat(
            tf_mat, self._tstamps, getattr(other, "_src", None), self._dst
        )

    def relative(self, other: "BaseSE3Trajectory") -> "NumpySE3Trajectory":
        """Per-frame ``other_i^{-1} @ self_i`` (``self`` in ``other``'s body frame);
        satisfies ``other.compose(self.relative(other)) == self``."""
        if len(other) != len(self):
            raise ValueError(f"relative length mismatch: {len(self)} vs {len(other)}")
        tf_mat = np.linalg.inv(other.transform_matrix()) @ self.transform_matrix()
        return type(self).from_tf_mat(tf_mat, self._tstamps, self._src, self._dst)

    def apply(self, points: Any) -> np.ndarray:
        """Transform 3D points by the trajectory.

        Shapes: ``(3,)`` or ``(M, 3)`` -> broadcast every frame to the same points,
        returning ``(N, 3)`` / ``(N, M, 3)``; ``(N, 3)`` -> one point per frame
        (paired) -> ``(N, 3)``; ``(N, M, 3)`` -> paired clouds -> ``(N, M, 3)``.
        """
        points = np.asarray(points, dtype=np.float64)
        R, t, num = self.rotation_matrix(), self._trans, len(self)
        if points.shape == (3,):  # single point, broadcast to all frames
            return R @ points + t
        if points.ndim == 2 and points.shape == (num, 3):  # one point per frame (paired)
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
        extrapolation)."""
        if self._tstamps is None:
            raise ValueError("trajectory has no timestamps; cannot interpolate")
        if len(self) < 2:
            raise ValueError("interpolate needs at least 2 frames")
        t_src = np.asarray(self._tstamps, dtype=np.float64)
        t_query = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if t_query.min() < t_src[0] - self._EPS or t_query.max() > t_src[-1] + self._EPS:
            raise ValueError("query timestamps out of range; no extrapolation")
        idx = np.clip(np.searchsorted(t_src, t_query, side="right") - 1, 0, len(self) - 2)
        t0, t1 = t_src[idx], t_src[idx + 1]
        alpha = np.where(t1 > t0, (t_query - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
        tf_mat = self.transform_matrix()
        poses = np.empty((t_query.shape[0], 7))
        for k, i in enumerate(idx):
            twist = self.log(np.linalg.inv(tf_mat[i]) @ tf_mat[i + 1])  # (6,)
            tf_k = tf_mat[i] @ self.exp(alpha[k] * twist)  # (4, 4)
            poses[k, :4] = Rotation.from_matrix(tf_k[:3, :3]).as_quat()
            poses[k, 4:] = tf_k[:3, 3]
        return type(self)(t_query, poses, self._src, self._dst, normalize=False)

    def align(
        self, other: "BaseSE3Trajectory", *, estimate_scale: bool = False
    ) -> "NumpySE3Trajectory":
        """Umeyama-align ``self`` onto ``other`` (over the matched per-frame
        positions) and return the aligned copy of ``self``. ``estimate_scale=False``
        is a rigid SE(3) fit; ``True`` adds a global scale (Sim(3)) for
        up-to-scale trajectories."""
        if len(other) != len(self):
            raise ValueError(f"align length mismatch: {len(self)} vs {len(other)}")
        src = self.translation()
        dst = np.asarray(other.transform_matrix()[:, :3, 3], dtype=np.float64)
        s, R, t = self._umeyama(src, dst, estimate_scale)
        R_new = R @ self.rotation_matrix()  # (N, 3, 3)
        t_new = s * (self._trans @ R.T) + t  # (N, 3)
        poses = np.concatenate([Rotation.from_matrix(R_new).as_quat(), t_new], axis=1)
        return type(self)(self._tstamps, poses, self._src, self._dst, normalize=False)

    # ------------------------------------------------------------------ #
    # Lie-group utilities (batched; accept single or (N, ...) inputs)
    # ------------------------------------------------------------------ #
    @staticmethod
    def log(se3_group: Any) -> np.ndarray:
        """SE(3) log: ``(N, 4, 4)`` / ``(4, 4)`` group -> ``(N, 6)`` / ``(6,)`` twist
        ``(rho, phi)`` (translation first)."""
        tf = np.asarray(se3_group, dtype=np.float64)
        single = tf.ndim == 2
        tf = tf[None] if single else tf
        phi = Rotation.from_matrix(tf[:, :3, :3]).as_rotvec()  # (N, 3)
        rho = np.einsum(
            "nij,nj->ni", NumpySE3Trajectory._left_jacobian_so3_inv(phi), tf[:, :3, 3]
        )
        xi = np.concatenate([rho, phi], axis=1)
        return xi[0] if single else xi

    @staticmethod
    def exp(se3_algebra: Any) -> np.ndarray:
        """SE(3) exp: ``(N, 6)`` / ``(6,)`` twist ``(rho, phi)`` (translation first)
        -> ``(N, 4, 4)`` / ``(4, 4)`` group."""
        xi = np.asarray(se3_algebra, dtype=np.float64)
        single = xi.ndim == 1
        xi = xi[None] if single else xi
        rho, phi = xi[:, :3], xi[:, 3:]
        tf = np.broadcast_to(np.eye(4), (xi.shape[0], 4, 4)).copy()
        tf[:, :3, :3] = Rotation.from_rotvec(phi).as_matrix()
        tf[:, :3, 3] = np.einsum(
            "nij,nj->ni", NumpySE3Trajectory._left_jacobian_so3(phi), rho
        )
        return tf[0] if single else tf

    @staticmethod
    def left_jacobian(se3_group: Any) -> np.ndarray:
        """SE(3) left Jacobian ``J_l`` for the ``(rho, phi)`` twist ordering.

        Accepts a group element ``(..., 4, 4)`` (logged first) or a twist
        ``(..., 6)``; returns ``(N, 6, 6)`` / ``(6, 6)``. Computed exactly via the
        Van Loan integral ``J_l = ∫_0^1 exp(s·ad) ds`` (the top-right block of
        ``expm([[ad, I], [0, 0]])``), valid for all twists with no small-angle case.
        """
        arr = np.asarray(se3_group, dtype=np.float64)
        xi = NumpySE3Trajectory.log(arr) if arr.shape[-2:] == (4, 4) else arr
        single = xi.ndim == 1
        xi = xi[None] if single else xi
        jac = np.empty((xi.shape[0], 6, 6))
        for k in range(xi.shape[0]):
            rho, phi = xi[k, :3], xi[k, 3:]
            ad = np.zeros((6, 6))
            ad[:3, :3] = NumpySE3Trajectory._skew(phi)
            ad[:3, 3:] = NumpySE3Trajectory._skew(rho)
            ad[3:, 3:] = NumpySE3Trajectory._skew(phi)
            aug = np.zeros((12, 12))
            aug[:6, :6], aug[:6, 6:] = ad, np.eye(6)
            jac[k] = expm(aug)[:6, 6:]
        return jac[0] if single else jac

    # ------------------------------------------------------------------ #
    # private so(3) / Umeyama math
    # ------------------------------------------------------------------ #
    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """``(3,)`` -> ``(3, 3)`` or ``(N, 3)`` -> ``(N, 3, 3)`` skew-symmetric ``[v]_x``."""
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1:
            x, y, z = v
            return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        K = np.zeros((v.shape[0], 3, 3))
        K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
        K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
        return K

    @classmethod
    def _left_jacobian_so3(cls, phi: np.ndarray) -> np.ndarray:
        """Batched SO(3) left Jacobian ``V(phi)`` -> ``(N, 3, 3)`` (the ``t = V @ rho``
        coupling in the SE(3) exp)."""
        phi = np.asarray(phi, dtype=np.float64).reshape(-1, 3)
        theta = np.linalg.norm(phi, axis=1)
        K = cls._skew(phi)
        eye = np.broadcast_to(np.eye(3), K.shape).copy()
        small = theta < cls._EPS
        a, b = np.full_like(theta, 0.5), np.full_like(theta, 1.0 / 6.0)
        th = theta[~small]
        a[~small] = (1.0 - np.cos(th)) / th**2
        b[~small] = (th - np.sin(th)) / th**3
        return eye + a[:, None, None] * K + b[:, None, None] * (K @ K)

    @classmethod
    def _left_jacobian_so3_inv(cls, phi: np.ndarray) -> np.ndarray:
        """Batched inverse SO(3) left Jacobian ``V(phi)^{-1}`` -> ``(N, 3, 3)`` (the
        ``rho = V^{-1} @ t`` coupling in the SE(3) log)."""
        phi = np.asarray(phi, dtype=np.float64).reshape(-1, 3)
        theta = np.linalg.norm(phi, axis=1)
        K = cls._skew(phi)
        eye = np.broadcast_to(np.eye(3), K.shape).copy()
        small = theta < cls._EPS
        c = np.full_like(theta, 1.0 / 12.0)
        th = theta[~small]
        c[~small] = 1.0 / th**2 - (1.0 + np.cos(th)) / (2.0 * th * np.sin(th))
        return eye - 0.5 * K + c[:, None, None] * (K @ K)

    @staticmethod
    def _umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool):
        """Least-squares similarity ``(s, R, t)`` mapping ``src`` onto ``dst``."""
        num = src.shape[0]
        mu_s, mu_d = src.mean(0), dst.mean(0)
        sc, dc = src - mu_s, dst - mu_d
        U, D, Vt = np.linalg.svd((dc.T @ sc) / num)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1, -1] = -1.0
        R = U @ S @ Vt
        s = (D * np.diag(S)).sum() / ((sc**2).sum() / num) if estimate_scale else 1.0
        return s, R, mu_d - s * (R @ mu_s)
