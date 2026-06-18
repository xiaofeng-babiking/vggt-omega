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

from vggt_omega.utils.rotation import mat_to_quat as _mat_to_quat, quat_to_mat as _quat_to_mat

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


# ----------------------------------------------------------------------------- #
# TorchSE3Pose — differentiable, device/dtype-aware backend on torch tensors
# ----------------------------------------------------------------------------- #


class TorchSE3Pose(BaseSE3Pose):
    """torch backend: stores ``(quat xyzw, translation)`` as tensors and keeps every
    op differentiable and device-aware. Reuses ``utils.rotation`` for the
    quaternion<->matrix conversions and hand-rolls the SE(3)-specific math. Inherits
    the derived surface (``relative`` / ``normalize`` / ``boxplus`` / ``boxminus`` /
    ``interpolate`` / operators) from :class:`BaseSE3Pose`.

    dtype policy: a tensor input keeps its own dtype/device (so float32 network
    outputs stay float32 with autograd intact); array-like input defaults to
    float64 for geometric precision. ``identity`` defaults to float64.
    """

    _EPS = 1e-8  # small-angle threshold for the so(3) Taylor branches

    def __init__(
        self,
        quat: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
        *,
        normalize: bool = True,
    ) -> None:
        q = self._to_tensor(quat).reshape(4)
        if not torch.is_floating_point(q):
            q = q.to(torch.float64)
        if normalize:
            q = q / torch.linalg.norm(q)
        self._quat = q
        if trans is None:
            self._trans = torch.zeros(3, dtype=q.dtype, device=q.device)
        else:
            self._trans = self._to_tensor(trans).reshape(3).to(dtype=q.dtype, device=q.device)

    # ----- construction -------------------------------------------------- #

    @classmethod
    def from_quat(
        cls,
        quat: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
        *,
        scalar_last: bool = True,
        normalize: bool = True,
    ) -> "TorchSE3Pose":
        q = cls._to_tensor(quat).reshape(4)
        if not scalar_last:  # wxyz -> xyzw
            q = q[[1, 2, 3, 0]]
        return cls(q, trans, normalize=normalize)

    @classmethod
    def from_rot_vec(cls, rot_vec: torch.Tensor, trans: Optional[torch.Tensor] = None) -> "TorchSE3Pose":
        return cls(cls._rotvec_to_quat(cls._to_tensor(rot_vec).reshape(3)), trans, normalize=False)

    @classmethod
    def from_rot_mat(cls, rot_mat: torch.Tensor, trans: Optional[torch.Tensor] = None) -> "TorchSE3Pose":
        return cls(_mat_to_quat(cls._to_tensor(rot_mat).reshape(3, 3)), trans, normalize=False)

    @classmethod
    def from_tf_mat(cls, tf_mat: torch.Tensor) -> "TorchSE3Pose":
        tf = cls._to_tensor(tf_mat)
        if tuple(tf.shape) not in ((4, 4), (3, 4)):
            raise ValueError(f"tf_mat must be (4, 4) or (3, 4), got {tuple(tf.shape)}")
        return cls.from_rot_mat(tf[:3, :3], tf[:3, 3])

    @classmethod
    def from_euler(
        cls,
        angles: torch.Tensor,
        seq: str,
        trans: Optional[torch.Tensor] = None,
        *,
        degrees: bool = False,
    ) -> "TorchSE3Pose":
        ang = cls._to_tensor(angles).reshape(-1)
        if degrees:
            ang = ang * (np.pi / 180.0)
        if ang.shape[0] != len(seq):
            raise ValueError(f"expected {len(seq)} angle(s) for seq '{seq}', got {ang.shape[0]}")
        unit = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
        quats = [
            cls._rotvec_to_quat(torch.tensor(unit[axis], dtype=ang.dtype, device=ang.device) * angle)
            for axis, angle in zip(seq.lower(), ang)
        ]
        if not seq.isupper():  # intrinsic (lowercase): moving axes -> reverse string order
            quats = quats[::-1]
        q = quats[0]
        for nxt in quats[1:]:
            q = cls._quat_mul(q, nxt)
        return cls(q, trans, normalize=True)

    @classmethod
    def identity(
        cls,
        *,
        like: Any = None,
        backend: Optional[str] = None,
        device: "Optional[Union[str, torch.device]]" = None,
        dtype: "Optional[Union[np.dtype, torch.dtype]]" = None,
    ) -> "TorchSE3Pose":
        if dtype is None:
            dtype = like.dtype if isinstance(like, torch.Tensor) else torch.float64
        if device is None and isinstance(like, torch.Tensor):
            device = like.device
        q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype, device=device)
        return cls(q, torch.zeros(3, dtype=dtype, device=device), normalize=False)

    @classmethod
    def exp(cls, tangent: torch.Tensor) -> "TorchSE3Pose":
        xi = cls._to_tensor(tangent).reshape(6)
        rho, phi = xi[:3], xi[3:]
        return cls(cls._rotvec_to_quat(phi), cls._left_jacobian_so3(phi) @ rho, normalize=False)

    # ----- accessors ----------------------------------------------------- #

    @property
    def quaternion(self) -> torch.Tensor:
        return self._quat.clone()

    @property
    def translation(self) -> torch.Tensor:
        return self._trans.clone()

    @property
    def rotation_matrix(self) -> torch.Tensor:
        return _quat_to_mat(self._quat)

    @property
    def transform_matrix(self) -> torch.Tensor:
        R = _quat_to_mat(self._quat)
        top = torch.cat([R, self._trans.reshape(3, 1)], dim=1)
        bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=R.dtype, device=R.device)
        return torch.cat([top, bottom], dim=0)

    # ----- core group ops ------------------------------------------------ #

    def inverse(self) -> "TorchSE3Pose":
        x, y, z, w = self._quat.unbind(-1)
        q_inv = torch.stack([-x, -y, -z, w])
        t_inv = -(_quat_to_mat(q_inv) @ self._trans)
        return type(self)(q_inv, t_inv, normalize=False)

    def compose(self, other: "BaseSE3Pose") -> "TorchSE3Pose":
        oq = self._to_tensor(other.quaternion).to(dtype=self._quat.dtype, device=self._quat.device)
        ot = self._to_tensor(other.translation).to(dtype=self._trans.dtype, device=self._trans.device)
        q = self._quat_mul(self._quat, oq)
        t = _quat_to_mat(self._quat) @ ot + self._trans
        return type(self)(q, t, normalize=False)

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        pts = self._to_tensor(points).to(dtype=self._quat.dtype, device=self._quat.device)
        return pts @ _quat_to_mat(self._quat).transpose(-1, -2) + self._trans

    # ----- Lie-group (backend math) -------------------------------------- #

    def log(self) -> torch.Tensor:
        phi = self._quat_to_rotvec(self._quat)
        rho = self._left_jacobian_so3_inv(phi) @ self._trans
        return torch.cat([rho, phi])

    def adjoint(self) -> torch.Tensor:
        R = _quat_to_mat(self._quat)
        tx = self._skew(self._trans)
        zero = torch.zeros((3, 3), dtype=R.dtype, device=R.device)
        top = torch.cat([R, tx @ R], dim=1)
        bottom = torch.cat([zero, R], dim=1)
        return torch.cat([top, bottom], dim=0)

    # ----- repr ---------------------------------------------------------- #

    def __repr__(self) -> str:
        q = self._quat.detach().cpu().numpy().round(4).tolist()
        t = self._trans.detach().cpu().numpy().round(4).tolist()
        return f"TorchSE3Pose(quat={q}, trans={t}, dtype={self._quat.dtype}, device={self._quat.device})"

    # ----- private torch helpers ----------------------------------------- #

    @staticmethod
    def _to_tensor(x) -> torch.Tensor:
        """Tensors pass through (dtype/device/grad preserved); array-like -> float64."""
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(np.asarray(x, dtype=np.float64))

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product of scalar-last quaternions (xyzw)."""
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        return torch.stack(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dim=-1,
        )

    @classmethod
    def _rotvec_to_quat(cls, rot_vec: torch.Tensor) -> torch.Tensor:
        """``so(3)`` axis-angle -> scalar-last unit quaternion."""
        theta = torch.linalg.norm(rot_vec)
        if theta < cls._EPS:  # half-angle Taylor: sin(θ/2)/θ -> 1/2, cos -> 1
            xyz = rot_vec * 0.5
            w = torch.ones((), dtype=rot_vec.dtype, device=rot_vec.device)
        else:
            half = 0.5 * theta
            xyz = rot_vec / theta * torch.sin(half)
            w = torch.cos(half)
        return torch.cat([xyz, w.reshape(1)])

    @classmethod
    def _quat_to_rotvec(cls, quat: torch.Tensor) -> torch.Tensor:
        """Scalar-last unit quaternion -> ``so(3)`` axis-angle (angle in [0, π])."""
        if quat[3] < 0:  # standardize to the w >= 0 hemisphere (shortest rotation)
            quat = -quat
        xyz, w = quat[:3], quat[3]
        nrm = torch.linalg.norm(xyz)
        if nrm < cls._EPS:  # small angle: rot_vec ≈ 2 * xyz
            return xyz * 2.0
        return xyz / nrm * (2.0 * torch.atan2(nrm, w))

    @staticmethod
    def _skew(v: torch.Tensor) -> torch.Tensor:
        """``(3,)`` -> ``(3, 3)`` skew-symmetric ``[v]_×`` (grad-preserving)."""
        x, y, z = v.unbind(-1)
        o = torch.zeros((), dtype=v.dtype, device=v.device)
        return torch.stack(
            [torch.stack([o, -z, y]), torch.stack([z, o, -x]), torch.stack([-y, x, o])]
        )

    @classmethod
    def _left_jacobian_so3(cls, phi: torch.Tensor) -> torch.Tensor:
        """SO(3) left Jacobian ``V(φ)`` (the SE(3) exp's ``t = V ρ`` coupling)."""
        theta = torch.linalg.norm(phi)
        K = cls._skew(phi)
        eye = torch.eye(3, dtype=phi.dtype, device=phi.device)
        if theta < cls._EPS:  # Taylor: I + 1/2 K + 1/6 K^2
            return eye + 0.5 * K + (K @ K) / 6.0
        a = (1.0 - torch.cos(theta)) / theta**2
        b = (theta - torch.sin(theta)) / theta**3
        return eye + a * K + b * (K @ K)

    @classmethod
    def _left_jacobian_so3_inv(cls, phi: torch.Tensor) -> torch.Tensor:
        """Inverse SO(3) left Jacobian ``V(φ)⁻¹`` (the SE(3) log coupling)."""
        theta = torch.linalg.norm(phi)
        K = cls._skew(phi)
        eye = torch.eye(3, dtype=phi.dtype, device=phi.device)
        if theta < cls._EPS:  # Taylor: I - 1/2 K + 1/12 K^2
            return eye - 0.5 * K + (K @ K) / 12.0
        c = 1.0 / theta**2 - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta))
        return eye - 0.5 * K + c * (K @ K)
