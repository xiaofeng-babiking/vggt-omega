"""Backend-agnostic SE(3) rigid-transform value type for VGGT-Omega.

:class:`BaseSE3Pose` is an **immutable value object for a single** rigid
transform in SE(3). It accepts every pose form the vendors emit (quaternion,
axis-angle, rotation matrix, 3x4/4x4 transform, Euler angles) and stores one
canonical representation, then exposes the full SE(3) / Lie-group operation
surface (``inverse``, ``compose`` / ``*`` / ``@``, ``apply``, ``exp`` / ``log``,
``adjoint``, ``interpolate``).

It models **one** pose — there is no batch dimension and there are no reduce
operations. A single pose still transforms an arbitrarily-shaped point cloud
through :meth:`apply`.

:class:`NumpySE3Pose` is the NumPy backend (CPU, float64, no autograd), built on
``scipy.spatial.transform.Rotation`` for the SO(3) parts.

Design
------
* **Unified representation = unit quaternion (scalar-last ``xyzw``) + translation**
  (logically ``(7,)``). A pose value type lives and dies by ``normalize`` /
  ``interpolate`` / ``exp`` / ``log``; those are cheap and numerically stable on
  a unit quaternion but awkward on a 3x3 ``R`` (which drifts off SO(3)). It is
  also 7 floats vs 12. The canonical ``(3, 4)`` w2c matrix the rest of the repo
  stores (``SequenceManifest.extrinsics``) is the top three rows of
  :attr:`transform_matrix` — a boundary conversion, not storage.

* **Backend-agnostic by inference, not by subclass.** Inputs may be NumPy *or*
  torch arrays; the backend is read off the input (mirroring
  ``geometry.closed_form_inverse_se3``) and preserved through every op — so the
  same class serves both the NumPy manifest/vendor layer and the differentiable
  torch training/model layer. The ABC therefore types array data as ``Any`` (the
  concrete backend decides). Array-free constructors (:meth:`identity`) take an
  explicit ``like=`` / ``backend=`` because there is nothing to infer from.

Conventions (locked)
--------------------
* Quaternion order is **scalar-last** ``[x, y, z, w]`` — matches
  ``utils.rotation`` and ``vendors.common.quat_to_rotation``.
* Active transform: a pose maps points as ``x' = R @ x + t``.
* ``a * b == a @ b == a.compose(b)`` — right-to-left, i.e. apply ``b`` first.
* ``~a == a.inverse()``; ``a(points) == a.apply(points)``.
* ``a - b == a.boxminus(b)`` — the ``⊟`` tangent ``log(b⁻¹ ∘ a)``.
* ``a.relative(b) == a ∘ b⁻¹`` — *a re-expressed in b's frame*, the same
  convention as ``geometry.compose_with_inverse`` (so ``a.relative(b) @ b == a``).
* Tangent / twist vectors are ``ξ = (ρ, φ)``, shape ``(6,)``, with the
  **translation part first** (``ρ`` = upper 3) and the ``so(3)`` part second
  (``φ`` = lower 3).
* The quaternion double cover (``q ≡ -q``) denotes the same rotation.

Implementation layering
-----------------------
The class is split like :class:`~vggt_omega.datasets.base_sequence.BaseSequence`:
a small **abstract backend kernel** (decorated ``@abstractmethod``) that a NumPy
or torch subclass must provide, plus a **derived surface** the base expresses in
terms of that kernel. A few derived methods that need raw matrix math
(:attr:`rotation_matrix`, :attr:`transform_matrix`, :meth:`adjoint`) are filled
in per backend rather than by the base.

Euler support uses ``scipy``-style sequence strings (``seq="xyz"`` intrinsic /
``"XYZ"`` extrinsic, ``degrees=`` flag), since ``scipy`` is already a dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation as _Rotation

# ----------------------------------------------------------------------------- #
# BaseSE3Pose — the abstract, backend-agnostic SE(3) value type
# ----------------------------------------------------------------------------- #


class BaseSE3Pose(ABC):
    """An immutable single SE(3) rigid transform.

    The canonical state is a unit quaternion (scalar-last ``xyzw``) plus a
    translation; there is no batch dimension. Every operation returns a **new**
    pose (or a plain array); nothing mutates in place. The numerical backend
    (NumPy or torch) is inferred from construction inputs and carried through
    unchanged — including autograd when the backend is torch.

    Subclasses implement only the methods marked ``@abstractmethod`` (the backend
    kernel); the remaining methods are provided by the base in terms of those.
    """

    # ===================================================================== #
    # Construction (canonical) — backend inferred from the input arrays
    # ===================================================================== #

    def __init__(
        self,
        quat: Any,
        trans: Optional[Any] = None,
        *,
        normalize: bool = True,
    ) -> None:
        """Build a pose from its canonical ``(quat, trans)`` state.

        Args:
            quat: ``(4,)`` scalar-last ``xyzw``. Need not be unit if ``normalize``
                is set.
            trans: ``(3,)``; defaults to zeros.
            normalize: re-normalise the quaternion to unit length on construction.
        """
        super().__init__()
        ...

    @classmethod
    @abstractmethod
    def from_quat(
        cls,
        quat: Any,
        trans: Optional[Any] = None,
        *,
        scalar_last: bool = True,
        normalize: bool = True,
    ) -> "BaseSE3Pose":
        """Construct from a quaternion (+ optional translation).

        Args:
            quat: ``(4,)``. Ordered ``xyzw`` if ``scalar_last`` else ``wxyz``.
            trans: ``(3,)`` or None (zeros).
            scalar_last: input quaternion order; output is always stored
                scalar-last.
            normalize: project to unit length.
        """
        ...

    @classmethod
    @abstractmethod
    def from_rot_vec(
        cls,
        rot_vec: Any,
        trans: Optional[Any] = None,
    ) -> "BaseSE3Pose":
        """Construct from an ``so(3)`` axis-angle (rotation) vector.

        Args:
            rot_vec: ``(3,)`` axis-angle — direction is the axis, magnitude the
                angle in radians (``R = exp_{so3}(rot_vec)``).
            trans: ``(3,)`` or None (zeros).
        """
        ...

    @classmethod
    @abstractmethod
    def from_rot_mat(
        cls,
        rot_mat: Any,
        trans: Optional[Any] = None,
    ) -> "BaseSE3Pose":
        """Construct from a ``(3, 3)`` rotation matrix (+ optional translation).

        ``rot_mat`` is assumed a proper rotation; pass through :meth:`normalize`
        first if it may have drifted off SO(3).
        """
        ...

    @classmethod
    @abstractmethod
    def from_tf_mat(cls, tf_mat: Any) -> "BaseSE3Pose":
        """Construct from a homogeneous transform.

        Args:
            tf_mat: ``(4, 4)`` or ``(3, 4)`` ``[R | t]``. The bottom row of a 4x4
                is ignored. This is the inverse of :attr:`transform_matrix` and the
                bridge to ``SequenceManifest.extrinsics`` (w2c, per-frame ``(3, 4)``).
        """
        ...

    @classmethod
    @abstractmethod
    def from_euler(
        cls,
        angles: Any,
        seq: str,
        trans: Optional[Any] = None,
        *,
        degrees: bool = False,
    ) -> "BaseSE3Pose":
        """Construct from Euler angles (``scipy``-style convention).

        Args:
            angles: ``(k,)`` with ``k == len(seq)`` (1-3 angles).
            seq: axis sequence, e.g. ``"xyz"`` (intrinsic, lowercase) or ``"XYZ"``
                (extrinsic, uppercase) — see ``scipy.spatial.transform.Rotation``.
            trans: ``(3,)`` or None (zeros).
            degrees: interpret ``angles`` as degrees instead of radians.
        """
        ...

    @classmethod
    @abstractmethod
    def identity(
        cls,
        *,
        like: Any = None,
        backend: Optional[str] = None,
        device: "Optional[Union[str, torch.device]]" = None,
        dtype: "Optional[Union[np.dtype, torch.dtype]]" = None,
    ) -> "BaseSE3Pose":
        """The identity pose.

        Being array-free, this needs an explicit backend: pass ``like=`` (an array
        or pose to inherit backend/device/dtype from) or ``backend=`` (``"numpy"``
        or ``"torch"``, + optional ``device`` / ``dtype``). Exactly one of
        ``like`` / ``backend`` is required.
        """
        ...

    # ===================================================================== #
    # Accessors — canonical state & derived matrices
    # ===================================================================== #

    @property
    @abstractmethod
    def quaternion(self) -> Any:
        """``(4,)`` unit quaternion, scalar-last ``xyzw``."""
        ...

    @property
    @abstractmethod
    def translation(self) -> Any:
        """``(3,)`` translation vector."""
        ...

    @property
    @abstractmethod
    def rotation_matrix(self) -> Any:
        """``(3, 3)`` rotation matrix (from the canonical quaternion)."""
        ...

    @property
    @abstractmethod
    def transform_matrix(self) -> Any:
        """``(4, 4)`` homogeneous transform ``[[R, t], [0, 1]]``. The repo's w2c
        extrinsics are its top three rows ``(3, 4)``."""
        ...

    # ===================================================================== #
    # Core SE(3) group operations
    # ===================================================================== #

    @abstractmethod
    def inverse(self) -> "BaseSE3Pose":
        """The inverse transform ``self⁻¹`` (rotation transposed, translation
        ``-Rᵀt``)."""
        ...

    @abstractmethod
    def compose(self, other: "BaseSE3Pose") -> "BaseSE3Pose":
        """Group product ``self ∘ other`` (apply ``other`` first, then ``self``).
        Also available as ``self * other`` / ``self @ other``."""
        ...

    @abstractmethod
    def apply(self, points: Any) -> Any:
        """Transform points: ``x' = R @ x + t``.

        Args:
            points: ``(..., 3)`` — one or many points; the single pose is applied
                to each.

        Returns:
            Transformed points, same shape as ``points``. Also ``self(points)``.
        """
        ...

    def relative(self, other: "BaseSE3Pose") -> "BaseSE3Pose":
        """``self ∘ other⁻¹`` — ``self`` re-expressed in ``other``'s frame.

        Matches ``geometry.compose_with_inverse``; satisfies
        ``self.relative(other) @ other == self``. With w2c camera poses and
        ``other`` the reference camera, this gives camera-from-reference."""
        return self.compose(other.inverse())

    def normalize(self) -> "BaseSE3Pose":
        """Return a copy with the quaternion re-projected to unit length (use after
        accumulating products or building from a drifted rotation matrix)."""
        return type(self).from_quat(self.quaternion, self.translation, normalize=True)

    # ===================================================================== #
    # Lie-group surface — anything touching the tangent space se(3)
    # ===================================================================== #

    @classmethod
    @abstractmethod
    def exp(cls, tangent: Any) -> "BaseSE3Pose":
        """The SE(3) exponential map: ``se(3)`` twist -> pose.

        Args:
            tangent: ``(6,)`` twist ``ξ = (ρ, φ)`` — translation part ``ρ`` first
                (upper 3), ``so(3)`` part ``φ`` second (lower 3). Backend is
                inferred from ``tangent``. Inverse of :meth:`log`.
        """
        ...

    @abstractmethod
    def log(self) -> Any:
        """SE(3) logarithm: ``(6,)`` twist ``ξ = (ρ, φ)`` (translation first).
        Inverse of :meth:`exp`."""
        ...

    @abstractmethod
    def adjoint(self) -> Any:
        """``(6, 6)`` adjoint ``Ad_self`` for the ``(ρ, φ)`` twist ordering
        (so ``self.compose(exp(ξ)) == exp(Ad_self @ ξ).compose(self)``)."""
        ...

    def boxplus(self, tangent: Any) -> "BaseSE3Pose":
        """Right ``⊞``: ``self ∘ exp(tangent)`` with ``tangent`` a ``(6,)`` twist.
        Inverse of :meth:`boxminus`."""
        return self.compose(type(self).exp(tangent))

    def boxminus(self, other: "BaseSE3Pose") -> Any:
        """Right ``⊟``: the ``(6,)`` twist with ``other.boxplus(result) == self``,
        i.e. ``log(other⁻¹ ∘ self)``."""
        return other.inverse().compose(self).log()

    def interpolate(self, other: "BaseSE3Pose", t: float) -> "BaseSE3Pose":
        """Constant-twist geodesic interpolation ``self ∘ exp(t · log(self⁻¹ ∘ other))``.

        ``t == 0`` -> ``self``, ``t == 1`` -> ``other``. Reduces to quaternion
        slerp on the rotation and lerp on the translation."""
        twist = self.inverse().compose(other).log()
        return self.compose(type(self).exp(t * twist))

    # ===================================================================== #
    # Operators (sugar over the methods above)
    # ===================================================================== #

    def __mul__(self, other: "BaseSE3Pose") -> "BaseSE3Pose":
        """``a * b`` -> :meth:`compose` (``a ∘ b``)."""
        return self.compose(other)

    def __matmul__(self, other: "BaseSE3Pose") -> "BaseSE3Pose":
        """``a @ b`` -> :meth:`compose` (identical to ``a * b``)."""
        return self.compose(other)

    def __invert__(self) -> "BaseSE3Pose":
        """``~a`` -> :meth:`inverse`."""
        return self.inverse()

    def __sub__(self, other: "BaseSE3Pose") -> Any:
        """``a - b`` -> :meth:`boxminus` (the ``⊟`` tangent ``log(b⁻¹ ∘ a)``)."""
        return self.boxminus(other)

    def __call__(self, points: Any) -> Any:
        """``a(points)`` -> :meth:`apply`."""
        return self.apply(points)

    @abstractmethod
    def __repr__(self) -> str: ...


# ----------------------------------------------------------------------------- #
# NumpySE3Pose — CPU / float64 backend on scipy.spatial.transform.Rotation
# ----------------------------------------------------------------------------- #


class NumpySE3Pose(BaseSE3Pose):
    """NumPy backend: stores ``(quat xyzw, translation)`` as ``float64`` arrays and
    delegates the SO(3) math to ``scipy.spatial.transform.Rotation``. CPU-only, no
    autograd. Inherits the derived surface (``relative`` / ``normalize`` /
    ``boxplus`` / ``boxminus`` / ``interpolate`` / operators) from
    :class:`BaseSE3Pose`."""

    _EPS: float = 1e-8

    def __init__(
        self,
        quat: np.ndarray,
        trans: Optional[np.ndarray] = None,
        *,
        normalize: bool = True,
    ) -> None:
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        if normalize:
            n = np.linalg.norm(q)
            if n < self._EPS:
                raise ValueError("quaternion has ~zero norm; cannot normalize")
            q = q / n
        self._quat = q
        self._trans = (
            np.zeros(3) if trans is None else np.asarray(trans, dtype=np.float64).reshape(3)
        )

    # ----- construction -------------------------------------------------- #

    @classmethod
    def from_quat(
        cls,
        quat: np.ndarray,
        trans: Optional[np.ndarray] = None,
        *,
        scalar_last: bool = True,
        normalize: bool = True,
    ) -> "NumpySE3Pose":
        q = np.asarray(quat, dtype=np.float64).reshape(4)
        if not scalar_last:  # wxyz -> xyzw
            q = q[[1, 2, 3, 0]]
        return cls(q, trans, normalize=normalize)

    @classmethod
    def from_rot_vec(cls, rot_vec: np.ndarray, trans: Optional[np.ndarray] = None) -> "NumpySE3Pose":
        q = _Rotation.from_rotvec(np.asarray(rot_vec, dtype=np.float64)).as_quat()
        return cls(q, trans, normalize=False)

    @classmethod
    def from_rot_mat(cls, rot_mat: np.ndarray, trans: Optional[np.ndarray] = None) -> "NumpySE3Pose":
        q = _Rotation.from_matrix(np.asarray(rot_mat, dtype=np.float64)).as_quat()
        return cls(q, trans, normalize=False)

    @classmethod
    def from_tf_mat(cls, tf_mat: np.ndarray) -> "NumpySE3Pose":
        tf = np.asarray(tf_mat, dtype=np.float64)
        if tf.shape not in ((4, 4), (3, 4)):
            raise ValueError(f"tf_mat must be (4, 4) or (3, 4), got {tf.shape}")
        return cls.from_rot_mat(tf[:3, :3], tf[:3, 3])

    @classmethod
    def from_euler(
        cls,
        angles: np.ndarray,
        seq: str,
        trans: Optional[np.ndarray] = None,
        *,
        degrees: bool = False,
    ) -> "NumpySE3Pose":
        q = _Rotation.from_euler(seq, angles, degrees=degrees).as_quat()
        return cls(q, trans, normalize=False)

    @classmethod
    def identity(
        cls,
        *,
        like: Any = None,
        backend: Optional[str] = None,
        device: "Optional[Union[str, torch.device]]" = None,
        dtype: "Optional[Union[np.dtype, torch.dtype]]" = None,
    ) -> "NumpySE3Pose":
        return cls(np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), normalize=False)

    @classmethod
    def exp(cls, tangent: np.ndarray) -> "NumpySE3Pose":
        xi = np.asarray(tangent, dtype=np.float64).reshape(6)
        rho, phi = xi[:3], xi[3:]
        q = _Rotation.from_rotvec(phi).as_quat()
        t = cls._left_jacobian_so3(phi) @ rho
        return cls(q, t, normalize=False)

    # ----- accessors ----------------------------------------------------- #

    @property
    def quaternion(self) -> np.ndarray:
        return self._quat.copy()

    @property
    def translation(self) -> np.ndarray:
        return self._trans.copy()

    @property
    def rotation_matrix(self) -> np.ndarray:
        return _Rotation.from_quat(self._quat).as_matrix()

    @property
    def transform_matrix(self) -> np.ndarray:
        tf = np.eye(4)
        tf[:3, :3] = _Rotation.from_quat(self._quat).as_matrix()
        tf[:3, 3] = self._trans
        return tf

    # ----- core group ops ------------------------------------------------ #

    def inverse(self) -> "NumpySE3Pose":
        r_inv = _Rotation.from_quat(self._quat).inv()
        return type(self)(r_inv.as_quat(), -r_inv.apply(self._trans), normalize=False)

    def compose(self, other: "BaseSE3Pose") -> "NumpySE3Pose":
        rot = _Rotation.from_quat(self._quat)
        other_q = np.asarray(other.quaternion, dtype=np.float64)
        other_t = np.asarray(other.translation, dtype=np.float64)
        q = (rot * _Rotation.from_quat(other_q)).as_quat()
        return type(self)(q, rot.apply(other_t) + self._trans, normalize=False)

    def apply(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        out = _Rotation.from_quat(self._quat).apply(pts.reshape(-1, 3)) + self._trans
        return out.reshape(pts.shape)

    # ----- Lie-group (backend math) -------------------------------------- #

    def log(self) -> np.ndarray:
        phi = _Rotation.from_quat(self._quat).as_rotvec()
        rho = self._left_jacobian_so3_inv(phi) @ self._trans
        return np.concatenate([rho, phi])

    def adjoint(self) -> np.ndarray:
        R = _Rotation.from_quat(self._quat).as_matrix()
        ad = np.zeros((6, 6))
        ad[:3, :3] = R
        ad[:3, 3:] = self._skew(self._trans) @ R
        ad[3:, 3:] = R
        return ad

    # ----- repr ---------------------------------------------------------- #

    def __repr__(self) -> str:
        q = np.array2string(self._quat, precision=4, suppress_small=True)
        t = np.array2string(self._trans, precision=4, suppress_small=True)
        return f"NumpySE3Pose(quat={q}, trans={t})"

    # ----- private so(3) math (SE(3) exp/log translation coupling V) ----- #

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        """``(3,)`` -> ``(3, 3)`` skew-symmetric ``[v]_×``."""
        x, y, z = v
        return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

    @classmethod
    def _left_jacobian_so3(cls, phi: np.ndarray) -> np.ndarray:
        """SO(3) left Jacobian ``V(φ)`` (the SE(3) exp's ``t = V ρ`` coupling)."""
        theta = float(np.linalg.norm(phi))
        K = cls._skew(phi)
        if theta < cls._EPS:  # Taylor: I + 1/2 K + 1/6 K^2
            return np.eye(3) + 0.5 * K + (K @ K) / 6.0
        a = (1.0 - np.cos(theta)) / theta**2
        b = (theta - np.sin(theta)) / theta**3
        return np.eye(3) + a * K + b * (K @ K)

    @classmethod
    def _left_jacobian_so3_inv(cls, phi: np.ndarray) -> np.ndarray:
        """Inverse SO(3) left Jacobian ``V(φ)⁻¹`` (the SE(3) log coupling)."""
        theta = float(np.linalg.norm(phi))
        K = cls._skew(phi)
        if theta < cls._EPS:  # Taylor: I - 1/2 K + 1/12 K^2
            return np.eye(3) - 0.5 * K + (K @ K) / 12.0
        c = 1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
        return np.eye(3) - 0.5 * K + c * (K @ K)
