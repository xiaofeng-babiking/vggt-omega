"""SE(3) trajectory frame sampling for VGGT-Omega.

:func:`sample_se3_trajectory` selects ``num`` frame indices from a sequence of
SE(3) poses so that the chosen frames are (approximately) **equally spaced in
SE(3) motion** rather than in index/time. "Motion" between two consecutive frames
is the body-twist magnitude ``‖log(Pₖ₋₁⁻¹ Pₖ)‖`` — the natural left-invariant
SE(3) distance, combining rotation and translation into one scalar. Accumulating
those gaps gives an arc-length parameterisation of the discrete trajectory; we
then place ``num`` targets at equal arc-length and snap each to the nearest frame.

This is *adaptive*: stretches with little motion (a near-static camera) contribute
little arc-length and are sampled sparsely, while fast-motion stretches are
sampled densely — i.e. redundant small-motion frames are skipped.

The function is deterministic for a fixed ``(start, end)`` window; randomness is
intended to come from the **caller** choosing/randomising that window.

:func:`sample_se3_random` is the stochastic sibling: same arc-length machinery,
but frames are drawn *randomly* without replacement with probability
proportional to the arc-length each frame owns, so spacing (and therefore
per-step motion magnitude) is arbitrary rather than equal, and the result
always has exactly ``num`` frames. Randomness comes from the global NumPy RNG.
"""

from typing import List, Tuple, Union

import numpy as np

from vggt_omega.datasets.sequences.base_sequence import BaseSequence


def _to_numpy(x) -> np.ndarray:
    """Backend-agnostic twist array -> NumPy float64 (NumPy or torch input)."""
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _resolve_window(n: int, num: int, start: int, end: int) -> Tuple[int, int, int]:
    """Resolve negative ``start``/``end`` (Python-slice style) against ``n`` and
    validate the request. Returns ``(start, end, window)`` with
    ``window = end - start + 1``. Shared by every SE(3) sampler in this module."""
    if n == 0:
        raise ValueError("seq is empty")
    if num < 1:
        raise ValueError(f"num must be >= 1, got {num}")
    start = start + n if start < 0 else start
    end = end + n if end < 0 else end
    if not (0 <= start < n):
        raise ValueError(f"start out of range: {start} (N={n})")
    if not (0 <= end < n):
        raise ValueError(f"end out of range: {end} (N={n})")
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")
    window = end - start + 1
    assert window >= num, f"window too small: end - start + 1 = {window} < num = {num}"
    return start, end, window


def _window_arc(seq, sensor, start: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame twist-norm gaps and cumulative arc-length over ``[start, end]``.

    ``gaps[j] = ‖log(P_{start+j}⁻¹ P_{start+j+1})‖`` (shape ``(window - 1,)``);
    ``cum[j]`` is the arc-length from ``start`` to frame ``start + j`` (shape
    ``(window,)``, ``cum[0] == 0.0``). Only the window slice is touched, so frames
    outside it never contribute."""
    window_twists = seq.get_poses(sensor)[start : end + 1].consecutive_twist()
    gaps = np.linalg.norm(_to_numpy(window_twists), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(gaps)])
    return gaps, cum


def sample_se3_trajectory(
    seq: BaseSequence,
    sensor: Union[int, str],
    num: int,
    start: int = 0,
    end: int = -1,
) -> Tuple[List[int], List[float]]:
    """Sample frame indices spaced ~uniformly in SE(3) arc-length.

    Per-frame gap is the SE(3) twist norm ``dₖ = ‖log(Pₖ₋₁⁻¹ Pₖ)‖``. The cumulative
    sum of these gaps over the ``[start, end]`` window is the arc-length; ``num``
    targets are placed at equal arc-length and each is snapped to the nearest frame
    (by cumulative arc-length), so dense-motion regions receive more samples than
    static ones.

    Poses come from ``seq.get_poses(sensor)`` — the precomputed per-sensor SE(3)
    trajectory (a :class:`~vggt_omega.datasets.utils.se3_trajectory.BaseSE3Trajectory`).
    The window ``[start, end]`` is sliced out and its consecutive body-twists are
    taken in one batched call (:meth:`consecutive_twist`); frames outside the window
    never affect the result.

    Args:
        seq: the :class:`BaseSequence` to sample from. ``seq.get_poses(sensor)`` must
            return an SE(3) trajectory positionally aligned with ``get_length`` (frame
            ``k`` is the ``k``-th element); the sampler operates on positional indices.
        sensor: the sensor id passed to ``seq.get_poses`` / ``seq.get_length``.
        num: number of equal-arc-length targets to place (``>= 1``). The returned
            list may be **shorter** than ``num`` because duplicate snapped indices
            are removed (e.g. when several targets fall on the same frame across a
            low-motion stretch).
        start: first frame index of the window (inclusive). Default ``0``.
        end: last frame index of the window (inclusive). Default ``-1`` (== the
            last frame); negative values index from the end like Python slicing.

    Returns:
        ``(indices, distances)`` of equal length:

        * ``indices``: the selected absolute frame indices into ``seq``,
          **strictly increasing** (duplicates removed), with ``indices[0] == start``
          and ``indices[-1] == end``.
        * ``distances``: the SE(3) **arc-length** to each kept frame from the
          previous kept frame — the sum of the per-frame twist-norm gaps
          ``Σ ‖log(Pₖ₋₁⁻¹ Pₖ)‖`` over the input frames strictly between the two
          kept indices (path length along the trajectory, not the direct chord
          twist between the two kept poses). ``distances[0] == 0.0``.

    Raises:
        ValueError: if ``num < 1``, the window is invalid, or the window is too
            small — requires ``end - start + 1 >= num``.
    """

    n_poses = seq.get_length(sensor)
    start, end, window = _resolve_window(n_poses, num, start, end)

    if num == 1 or start == end:
        return [start], [0.0]

    # Per-frame SE(3) motion gaps over the window; cum[j] is the arc-length from
    # `start` to frame start+j.
    gaps, cum = _window_arc(seq, sensor, start, end)
    total = float(cum[-1])

    if total <= 0.0:
        # No motion in the window: spread evenly by index instead.
        picks = np.linspace(start, end, num)
        snapped = [int(round(p)) for p in picks]
    else:
        # Place num targets at equal arc-length, snap each to the nearest frame.
        targets = np.linspace(0.0, total, num)
        nearest = np.abs(cum[None, :] - targets[:, None]).argmin(axis=1)
        snapped = [start + int(j) for j in nearest]

    # Pin the window endpoints exactly, then keep strictly-increasing unique indices
    # (duplicate snaps across a low-motion stretch collapse to one frame).
    snapped[0], snapped[-1] = start, end
    indices: List[int] = []
    for i in snapped:
        if not indices or i > indices[-1]:
            indices.append(i)
    if indices[-1] != end:  # ensure the endpoint survives de-duplication
        indices.append(end)

    # distances[k] = SE(3) arc-length from the previous kept frame to this one
    # (== Σ per-frame twist norms between consecutive kept indices); distances[0]=0.
    distances = [0.0]
    for k in range(1, len(indices)):
        distances.append(float(cum[indices[k] - start] - cum[indices[k - 1] - start]))
    return indices, distances


# Tiny uniform probability floor mixed into the motion weights. It keeps every
# frame drawable, so the without-replacement draw can always fill ``num`` slots
# even when fewer than ``num`` frames carry motion (motion frames are drawn
# first, static stretches fill the remainder ~uniformly).
LAMBDA_FLOOR = 1e-3


def sample_se3_random(
    seq: BaseSequence,
    sensor: Union[int, str],
    num: int,
    start: int = 0,
    end: int = -1,
) -> Tuple[List[int], List[float]]:
    """Randomly sample ``num`` frames with probability ~ local SE(3) motion.

    The stochastic sibling of :func:`sample_se3_trajectory`: instead of placing
    frames at *equal* arc-length, it draws ``num`` distinct frames **without
    replacement** with per-frame probability proportional to the arc-length each
    frame owns on the window's arc-length axis (half of each adjacent twist-norm
    gap — its Voronoi cell). Statistically this is a uniform random draw on the
    arc-length axis snapped to frames, but collisions are impossible, so the
    result always has **exactly** ``num`` indices. Inter-frame motion magnitudes
    are therefore *arbitrary* (from near-zero to large) while static stretches
    remain under-sampled — the regime wanted for joint 3DGS self-render
    training.

    Unlike :func:`sample_se3_trajectory`, the window endpoints are **not**
    pinned, and the result is random per call: randomness comes from the global
    NumPy RNG (seed ``np.random`` for reproducibility), on top of whatever
    window randomisation the caller applies.

    Args:
        seq: the :class:`BaseSequence` to sample from; ``seq.get_poses(sensor)``
            must be positionally aligned with ``get_length``.
        sensor: the sensor id passed to ``seq.get_poses`` / ``seq.get_length``.
        num: number of distinct frames to draw (``>= 1``); the result always has
            exactly ``num`` indices. Even ``num == 1`` is a weighted random draw
            (only a single-frame window ``start == end`` is deterministic).
        start: first frame index of the window (inclusive). Default ``0``;
            negative values index from the end, Python-slice style.
        end: last frame index of the window (inclusive). Default ``-1`` (the
            last frame); negative values index from the end.

    Returns:
        ``(indices, distances)`` of equal length ``num``:

        * ``indices``: the selected absolute frame indices, **strictly
          increasing**, each within ``[start, end]``.
        * ``distances``: the SE(3) **path arc-length** to each kept frame from
          the previous kept frame (sum of per-frame twist norms between them),
          ``distances[0] == 0.0`` — same semantics as
          :func:`sample_se3_trajectory`.

    Raises:
        ValueError: if the sequence is empty, ``num < 1``, or the window is
            invalid (out-of-range / ``end < start``).
        AssertionError: if ``end - start + 1 < num``.
    """
    n_poses = seq.get_length(sensor)
    start, end, window = _resolve_window(n_poses, num, start, end)

    if start == end:
        return [start], [0.0]

    gaps, cum = _window_arc(seq, sensor, start, end)
    total = float(cum[-1])

    if total <= 0.0:
        # Fully static window: motion carries no signal -> uniform draw
        # (degrades to sample_frame_indices behaviour).
        p = None
    else:
        # Each frame owns half of each adjacent gap (its Voronoi cell on the
        # arc-length axis); w sums to `total` before the floor.
        w = np.zeros(window)
        w[:-1] += gaps / 2.0
        w[1:] += gaps / 2.0
        w += LAMBDA_FLOOR * total / window
        p = w / w.sum()

    picks = np.random.choice(window, size=num, replace=False, p=p)
    indices = sorted(start + int(i) for i in picks)

    distances = [0.0] + [
        float(cum[indices[k] - start] - cum[indices[k - 1] - start])
        for k in range(1, len(indices))
    ]
    return indices, distances
