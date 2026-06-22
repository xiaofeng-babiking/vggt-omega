"""Backend-agnostic continuous-time SE(3) trajectory for VGGT-Omega.

A :class:`BaseSE3Trajectory` is an **ordered collection of timestamped SE(3)
poses** (built on :class:`~vggt_omega.datasets.se3_pose.BaseSE3Pose`) plus the
operations a trajectory — as opposed to a single pose — needs: fitting a smooth
continuous-time curve through the samples (:meth:`fit`), evaluating that curve at
an arbitrary time (:meth:`interpolate`), re-sampling it uniformly over a
sub-interval (:meth:`sample`), rigidly/similarity transforming the whole path
(:meth:`transform`), and aligning one trajectory onto another with Umeyama
(:meth:`align`).

Spline model
------------
:meth:`fit` builds a **cumulative cubic B-spline on SE(3)** (Kim, Kim & Shin 1995;
Sommer et al., CVPR 2020). This is the standard continuous-time representation for
visual / inertial odometry: it is :math:`C^2`-continuous, has local support (a
sample only influences a small time window), and — unlike a naive sequence of
pairwise geodesics — is smooth in *both* rotation and translation.

A segment between two consecutive control poses is, for local time
:math:`u \\in [0, 1]`,

.. math::

    T(u) = T_{i-1} \\prod_{j=1}^{3}
           \\exp\\!\\big(\\tilde B_j(u)\\, \\Omega_{i+j-1}\\big),
    \\qquad \\Omega_k = \\log\\!\\big(T_{k-1}^{-1} T_k\\big),

where :math:`\\tilde B_j` are the **cumulative** uniform-cubic-B-spline basis
functions (rows of :data:`_CUMULATIVE_BASIS` applied to
:math:`[1, u, u^2, u^3]`). A plain cubic B-spline does *not* pass through its
control poses; we make the curve **interpolate the first and last input pose** by
triple-padding the endpoints (two extra copies on each side — the standard
"clamped" knot trick). The interior poses are approximated, not interpolated,
which is exactly what gives the curve its smoothness.

Parameterisation
----------------
Each input pose carries a time. Times default to a uniform grid on ``[0, 1]`` but
may be given explicitly (timestamps, frame indices, anything monotonically
increasing); they are stored and rescaled to ``[0, 1]`` internally so
:meth:`interpolate` accepts ``t`` in ``[0, 1]`` regardless of the original units.

Design / layering
-----------------
Mirrors :mod:`vggt_omega.datasets.se3_pose`: a small **abstract backend kernel**
(``@abstractmethod`` — the array primitives that differ between NumPy and torch)
plus a **derived surface** the base expresses purely in terms of that kernel and
the (already backend-agnostic) ``BaseSE3Pose`` operations. The bulk of the spline
/ Umeyama logic therefore lives once, in the base, and the two concrete backends
(:class:`NumpySE3Trajectory`, :class:`TorchSE3Trajectory`) only provide stacking /
linear-algebra primitives. The backend is inferred from the poses' own backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose, TorchSE3Pose

# Cumulative uniform cubic B-spline basis: rows are the cumulative basis functions
# B̃_1, B̃_2, B̃_3 (the 0-th basis is identically 1 and multiplies the segment's base
# pose, so it is dropped). Each row is the polynomial coefficients in
# [1, u, u², u³]; B̃_j(u) = row_j · [1, u, u², u³]. (Qin 1998; Sommer et al. 2020.)
_CUMULATIVE_BASIS: Tuple[Tuple[float, float, float, float], ...] = (
    (5.0 / 6.0, 0.5, -0.5, 1.0 / 6.0),
    (1.0 / 6.0, 0.5, 0.5, -1.0 / 3.0),
    (0.0, 0.0, 0.0, 1.0 / 6.0),
)


# ----------------------------------------------------------------------------- #
# BaseSE3Trajectory — abstract, backend-agnostic continuous-time SE(3) curve
# ----------------------------------------------------------------------------- #


class BaseSE3Trajectory(ABC):
    """An ordered, timestamped sequence of SE(3) poses with a fittable spline.

    State is the list of control poses (:attr:`poses`) and their times
    (:attr:`times`, normalised to ``[0, 1]``). :meth:`fit` precomputes the
    cumulative-B-spline data so that :meth:`interpolate` / :meth:`sample` are
    cheap; the curve is fit lazily on first evaluation if :meth:`fit` was not
    called explicitly. The backend (NumPy or torch) is inferred from the poses and
    carried through every operation.

    Subclasses implement only the ``@abstractmethod`` kernel; the spline / Umeyama
    surface is provided by the base.
    """

    _MIN_POSES: int = 2

    def __init__(self, se3_poses: Sequence[BaseSE3Pose], times: Optional[Sequence[float]] = None):
        """Initialize an SE(3) trajectory from poses (and optional times).

        Args:
            se3_poses: ordered poses, all of the same backend; ``len >= 2``.
            times: per-pose times, strictly increasing. ``None`` -> a uniform grid
                on ``[0, 1]``. Any monotone units are accepted and rescaled to
                ``[0, 1]`` internally.
        """
        poses = list(se3_poses)
        if len(poses) < self._MIN_POSES:
            raise ValueError(f"need at least {self._MIN_POSES} poses, got {len(poses)}")
        pose_type = self._pose_type()
        if not all(isinstance(p, pose_type) for p in poses):
            raise TypeError(f"all poses must be {pose_type.__name__} for this backend")
        self._poses: List[BaseSE3Pose] = poses
        self._times = self._normalize_times(times, len(poses))
        # Lazily-built cumulative-B-spline state (see :meth:`fit`).
        self._control: Optional[List[BaseSE3Pose]] = None
        self._omega: Optional[List[Any]] = None  # Ω_k = log(C_{k-1}⁻¹ C_k) per control gap

    # ===================================================================== #
    # Abstract backend kernel — the array primitives that differ per backend
    # ===================================================================== #

    @classmethod
    @abstractmethod
    def _pose_type(cls) -> type:
        """The concrete ``BaseSE3Pose`` subclass this trajectory backend uses."""
        ...

    @abstractmethod
    def _asarray(self, x: Any) -> Any:
        """Coerce a Python/array-like to this backend's array (float)."""
        ...

    @staticmethod
    @abstractmethod
    def _to_numpy(x: Any) -> np.ndarray:
        """This backend's array -> NumPy (for shared math: Umeyama, knot search)."""
        ...

    @abstractmethod
    def _stack_translations(self) -> Any:
        """``(N, 3)`` stack of the control-pose translations, this backend."""
        ...

    # ===================================================================== #
    # Accessors
    # ===================================================================== #

    @property
    def poses(self) -> List[BaseSE3Pose]:
        """The input control poses (a copy of the list)."""
        return list(self._poses)

    @property
    def times(self) -> Any:
        """``(N,)`` per-pose times, normalised to ``[0, 1]``."""
        return self._asarray(self._times)

    def __len__(self) -> int:
        return len(self._poses)

    # ===================================================================== #
    # fit — build the cumulative cubic B-spline (clamped / endpoint-interpolating)
    # ===================================================================== #

    def fit(self) -> "BaseSE3Trajectory":
        """Fit the trajectory as a cumulative cubic B-spline on SE(3).

        Triple-pads the endpoints (clamped knots) so the curve interpolates the
        first and last input pose, then precomputes the per-gap tangents
        ``Ω_k = log(C_{k-1}⁻¹ C_k)`` used by :meth:`interpolate`. Returns ``self``
        so it can be chained. Called automatically on first evaluation if omitted.
        """
        pad = 2  # cubic: two extra copies of each endpoint => curve hits the ends
        control = [self._poses[0]] * pad + list(self._poses) + [self._poses[-1]] * pad
        omega: List[Any] = []
        for k in range(1, len(control)):
            # Ω_k = log(C_{k-1}⁻¹ ∘ C_k); boxminus(other)=log(other⁻¹∘self), so the
            # base C_{k-1} must be the *argument*: C_k.boxminus(C_{k-1}).
            omega.append(control[k].boxminus(control[k - 1]))
        self._control = control
        self._omega = omega
        return self

    def _ensure_fit(self) -> None:
        if self._control is None:
            self.fit()

    # ===================================================================== #
    # interpolate / sample — evaluate the fitted curve
    # ===================================================================== #

    def interpolate(self, t: float) -> BaseSE3Pose:
        """Evaluate the spline at normalised time ``t`` in ``[0, 1]``.

        ``t == 0`` -> first pose, ``t == 1`` -> last pose. ``t`` is mapped to the
        control-pose index grid, the enclosing cubic segment is selected, and the
        cumulative blend ``C_{i-1} · ∏_j exp(B̃_j(u) Ω)`` is accumulated.
        """
        self._ensure_fit()
        assert self._control is not None and self._omega is not None
        t = float(np.clip(t, 0.0, 1.0))

        # Map t in [0,1] to the global segment parameter g in [0, n_seg], where
        # segment i (1-based) spans g in [i-1, i] with local u = g-(i-1). The clamped
        # padding makes g=0 -> first pose and g=n_seg -> last pose. Non-uniform input
        # times warp t->g through the stored knot grid so spacing is honoured.
        n_seg = len(self._control) - 3  # = N + 1 for N input poses
        knots_t = self._times  # (N,) normalised times of the input poses, in [0,1]
        # segment-parameter knots for the N input poses: 0, 1, ..., n_seg, sampled at
        # the same fractional positions the poses sit at (endpoints pinned to 0/n_seg).
        knots_g = np.linspace(0.0, float(n_seg), len(knots_t))
        g = float(np.interp(t, knots_t, knots_g))
        i = int(np.clip(int(np.floor(g)) + 1, 1, n_seg))
        u = float(np.clip(g - (i - 1), 0.0, 1.0))

        pose = self._control[i - 1]
        u_powers = (1.0, u, u * u, u * u * u)
        for j in range(3):
            b = sum(c * p for c, p in zip(_CUMULATIVE_BASIS[j], u_powers))
            omega = self._omega[i + j - 1]  # Ω for gap (C_{i+j-1}, C_{i+j})
            pose = pose.boxplus(b * omega)
        return pose

    def sample(self, start: float, end: float, n: int) -> List[BaseSE3Pose]:
        """Uniformly sample ``n`` poses over the sub-interval ``[start, end]``.

        ``start`` / ``end`` are normalised times in ``[0, 1]`` (the same scale as
        :meth:`interpolate`); they are clamped to ``[0, 1]``. ``n == 1`` returns the
        single pose at ``start``; otherwise the ``n`` samples are taken at
        ``t = start, …, end`` inclusive of both ends. Use ``sample(0.0, 1.0, n)`` to
        cover the whole trajectory.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        start = float(np.clip(start, 0.0, 1.0))
        end = float(np.clip(end, 0.0, 1.0))
        if n == 1:
            return [self.interpolate(start)]
        return [self.interpolate(start + (end - start) * k / (n - 1)) for k in range(n)]

    # ===================================================================== #
    # transform — push the whole trajectory through an SE(3)/SIM(3) map
    # ===================================================================== #

    def transform(self, se3_tf: BaseSE3Pose, scale: float = 1.0) -> "BaseSE3Trajectory":
        """Left-apply a similarity ``(scale, se3_tf)`` to every pose.

        Maps each pose ``T`` to ``T' = S ∘ T`` with the rotation taken from
        ``se3_tf`` and the translation scaled: ``t' = scale · (R_tf t + t_tf)``.
        Returns a **new** trajectory (times preserved). ``scale == 1`` is a plain
        rigid SE(3) transform; ``scale != 1`` is SIM(3) (used after
        :meth:`align`)."""
        new_poses = [self._compose_sim3(se3_tf, p, scale) for p in self._poses]
        return type(self)(new_poses, self._times)

    def _compose_sim3(self, tf: BaseSE3Pose, pose: BaseSE3Pose, scale: float) -> BaseSE3Pose:
        composed = tf.compose(pose)  # rotation & rigid translation
        if scale == 1.0:
            return composed
        pose_type = self._pose_type()
        return pose_type.from_quat(composed.quaternion, scale * composed.translation)

    # ===================================================================== #
    # align — Umeyama similarity alignment of one trajectory onto another
    # ===================================================================== #

    def align(
        self, other: "BaseSE3Trajectory", with_scale: bool = True
    ) -> Tuple["BaseSE3Trajectory", BaseSE3Pose, float]:
        """Align ``self`` onto ``other`` by the Umeyama (1991) similarity transform.

        Solves for ``(scale, R, t)`` minimising
        ``Σ ‖ scale·R·p_i + t − q_i ‖²`` between this trajectory's positions
        ``p_i`` and ``other``'s ``q_i`` (the two must have equal length). With
        ``with_scale=False`` the scale is fixed to 1 (rigid SE(3) alignment, as in
        common ATE evaluation).

        Returns ``(aligned_trajectory, transform_pose, scale)`` — the aligned copy
        plus the recovered similarity, so callers can reuse the transform.
        """
        if len(self) != len(other):
            raise ValueError(f"trajectories must match in length: {len(self)} vs {len(other)}")
        src = self._to_numpy(self._stack_translations())
        dst = other._to_numpy(other._stack_translations())
        R, t, scale = _umeyama(src, dst, with_scale=with_scale)

        pose_type = self._pose_type()
        tf = pose_type.from_rot_mat(self._asarray(R), self._asarray(t))
        return self.transform(tf, scale=scale), tf, scale

    # ===================================================================== #
    # shared helpers
    # ===================================================================== #

    @staticmethod
    def _normalize_times(times: Optional[Sequence[float]], n: int) -> np.ndarray:
        """Validate / rescale times to a strictly-increasing grid on ``[0, 1]``."""
        if times is None:
            return np.linspace(0.0, 1.0, n)
        ts = np.asarray(times, dtype=np.float64).reshape(-1)
        if ts.shape[0] != n:
            raise ValueError(f"expected {n} times, got {ts.shape[0]}")
        if np.any(np.diff(ts) <= 0):
            raise ValueError("times must be strictly increasing")
        span = ts[-1] - ts[0]
        if span <= 0:
            raise ValueError("times must span a positive interval")
        return (ts - ts[0]) / span

    def __repr__(self) -> str:
        fitted = "fitted" if self._control is not None else "unfitted"
        return f"{type(self).__name__}(n={len(self)}, {fitted})"

def _umeyama(
    src: np.ndarray, dst: np.ndarray, *, with_scale: bool
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Umeyama least-squares similarity ``dst ≈ scale·R·src + t``.

    Args:
        src: ``(N, 3)`` source points.
        dst: ``(N, 3)`` target points.
        with_scale: solve for scale; else fix ``scale = 1``.

    Returns ``(R (3,3), t (3,), scale)`` — all NumPy float64. Reflection-safe (a
    det-correcting sign flip keeps ``R`` a proper rotation).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:  # ensure a proper rotation
        S[2, 2] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c**2).sum() / n
        scale = float((D * np.diag(S)).sum() / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0
    t = mu_dst - scale * R @ mu_src
    return R, t, scale


# ----------------------------------------------------------------------------- #
# NumpySE3Trajectory — CPU / float64 backend
# ----------------------------------------------------------------------------- #


class NumpySE3Trajectory(BaseSE3Trajectory):
    """NumPy backend trajectory over :class:`NumpySE3Pose` (CPU, float64)."""

    @classmethod
    def _pose_type(cls) -> type:
        return NumpySE3Pose

    @staticmethod
    def _asarray(x: Any) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def _stack_translations(self) -> np.ndarray:
        return np.stack([p.translation for p in self._poses], axis=0)


# ----------------------------------------------------------------------------- #
# TorchSE3Trajectory — differentiable, device/dtype-aware backend
# ----------------------------------------------------------------------------- #


class TorchSE3Trajectory(BaseSE3Trajectory):
    """torch backend trajectory over :class:`TorchSE3Pose`.

    Operations stay differentiable and device-aware via the underlying poses; the
    Umeyama solve runs on CPU NumPy (it is a one-off fit, not a differentiated op)
    and its result is mapped back to the poses' dtype/device.
    """

    @classmethod
    def _pose_type(cls) -> type:
        return TorchSE3Pose

    def _ref(self) -> torch.Tensor:
        """A reference tensor to inherit dtype/device from (the first quaternion)."""
        return self._poses[0].quaternion

    def _asarray(self, x: Any) -> torch.Tensor:  # type: ignore[override]
        ref = self._ref()
        if isinstance(x, torch.Tensor):
            return x.to(dtype=ref.dtype, device=ref.device)
        return torch.as_tensor(np.asarray(x, dtype=np.float64)).to(
            dtype=ref.dtype, device=ref.device
        )

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    def _stack_translations(self) -> torch.Tensor:
        return torch.stack([p.translation for p in self._poses], dim=0)
