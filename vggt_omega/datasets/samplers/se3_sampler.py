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
"""

from typing import List, Tuple, Union

import numpy as np

from vggt_omega.datasets.sequences.base_sequence import BaseSequence


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

    def _to_numpy(x) -> np.ndarray:
        """Backend-agnostic twist array -> NumPy float64 (NumPy or torch input)."""
        if hasattr(x, "detach"):  # torch.Tensor
            return x.detach().cpu().numpy().astype(np.float64)
        return np.asarray(x, dtype=np.float64)

    n_poses = seq.get_length(sensor)
    if n_poses == 0:
        raise ValueError("seq is empty")
    if num < 1:
        raise ValueError(f"num must be >= 1, got {num}")

    # Resolve negative indices (Python-slicing style) and validate the window.
    start = start + n_poses if start < 0 else start
    end = end + n_poses if end < 0 else end
    if not (0 <= start < n_poses):
        raise ValueError(f"start out of range: {start} (N={n_poses})")
    if not (0 <= end < n_poses):
        raise ValueError(f"end out of range: {end} (N={n_poses})")
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")

    window = end - start + 1
    assert window >= num, f"window too small: end - start + 1 = {window} < num = {num}"

    if num == 1 or start == end:
        return [start], [0.0]

    # Per-frame SE(3) motion gaps over the window. ``consecutive_twist`` on the
    # sliced trajectory returns ``log(P_i⁻¹ P_{i+1})`` for each consecutive pair
    # (shape ``(window - 1, 6)``); cum[j] is the arc-length from `start` to frame
    # start+j. Only the window slice is touched, so frames outside it never
    # contribute.
    window_twists = seq.get_poses(sensor)[start : end + 1].consecutive_twist()
    gaps = np.linalg.norm(_to_numpy(window_twists), axis=1)  # (window - 1,)
    cum = np.concatenate([[0.0], np.cumsum(gaps)])  # (window,)
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
