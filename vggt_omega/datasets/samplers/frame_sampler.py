"""Lightweight random frame-index sampling for VGGT-Omega.

:func:`sample_frame_indices` is the cheap, pose-free cousin of
:func:`~vggt_omega.datasets.samplers.se3_sampler.sample_se3_trajectory`: it draws
``num`` *distinct* frame indices **uniformly at random** from a ``[start, end]``
window and returns them sorted. No poses are read (only ``seq.get_length``), so it
costs nothing beyond a NumPy ``choice`` — use it when motion-adaptive arc-length
sampling is unnecessary and a plain random subset of frames will do.

It deliberately mirrors ``sample_se3_trajectory``'s signature and
``(indices, distances)`` return contract so the two are drop-in interchangeable;
here "distance" is simply the per-step **index gap** (``idx[k] - idx[k-1]``) rather
than SE(3) arc-length. Randomness comes from the global NumPy RNG, so callers seed
``np.random`` for reproducibility (as the dataset pipeline already does elsewhere).
"""

from typing import List, Tuple, Union

import numpy as np

from vggt_omega.datasets.sequences.base_sequence import BaseSequence


def sample_frame_indices(
    seq: BaseSequence,
    sensor: Union[int, str],
    num: int,
    start: int = 0,
    end: int = -1,
) -> Tuple[List[int], List[float]]:
    """Randomly sample ``num`` distinct frame indices from the ``[start, end]`` window.

    ``num`` distinct indices are drawn uniformly without replacement from the
    inclusive window ``[start, end]`` and returned **sorted ascending** (temporal
    order). The window length is read per-sensor via ``seq.get_length(sensor)``;
    no frames are decoded and no poses are touched.

    Args:
        seq: the :class:`BaseSequence` to sample from. Only ``seq.get_length(sensor)``
            is used (the window size); pixels/poses are never read.
        sensor: the sensor id passed to ``seq.get_length`` (one synced timeline).
        num: number of distinct frames to draw (``>= 1``). The result always has
            exactly ``num`` indices (no de-duplication, unlike the SE(3) sampler).
        start: first frame index of the window (inclusive). Default ``0``. Negative
            values index from the end, Python-slice style.
        end: last frame index of the window (inclusive). Default ``-1`` (the last
            frame); negative values index from the end, Python-slice style.

    Returns:
        ``(indices, distances)`` of equal length ``num``:

        * ``indices``: the selected absolute frame indices into ``seq``, **strictly
          increasing** (distinct + sorted), each within ``[start, end]``.
        * ``distances``: the per-step **index gap** to each frame from the previous
          one, ``distances[k] = float(indices[k] - indices[k-1])``, with
          ``distances[0] == 0.0``. (Mirrors the SE(3) sampler's distance slot; here
          it measures index spacing, not motion.)

    Raises:
        ValueError: if the sequence is empty, ``num < 1``, or the window is invalid
            (out-of-range / ``end < start``).
        AssertionError: if the window is too small — requires ``end - start + 1 >= num``
            (cannot draw ``num`` distinct frames from fewer).
    """
    n = seq.get_length(sensor)
    if n == 0:
        raise ValueError("seq is empty")
    if num < 1:
        raise ValueError(f"num must be >= 1, got {num}")

    # Resolve negative indices (Python-slicing style) and validate the window.
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

    if num == 1 or start == end:
        return [start], [0.0]

    # Uniform distinct draw over the inclusive window, returned in temporal order.
    picks = np.random.choice(np.arange(start, end + 1), size=num, replace=False)
    indices = sorted(int(i) for i in picks)

    # distances[k] = index gap from the previous kept frame; distances[0] = 0.0.
    distances = [0.0] + [
        float(indices[k] - indices[k - 1]) for k in range(1, len(indices))
    ]
    return indices, distances
