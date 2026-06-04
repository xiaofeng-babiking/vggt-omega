"""Generic, vendor-agnostic helpers shared across dataset vendors.

These are pure utilities with no dataset-specific constants or conventions:
timestamp-keyed index parsing, greedy nearest-timestamp association, and
quaternion math. Vendor-specific logic (pose conventions, depth encodings,
intrinsics tables) lives on the individual vendor datasets instead.
"""
from __future__ import annotations

import numpy as np


def read_file_list(path: str) -> dict[float, list[str]]:
    """Parse an index file ('timestamp v1 v2 ...') -> {timestamp: [v1, v2, ...]}."""
    out: dict[float, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            out[float(parts[0])] = parts[1:]
    return out


def associate(first, second, max_diff: float = 0.02, offset: float = 0.0):
    """Greedy nearest-timestamp matching (TUM associate.py). Returns sorted [(t1, t2)]."""
    potential = [
        (abs(a - (b + offset)), a, b)
        for a in first
        for b in second
        if abs(a - (b + offset)) < max_diff
    ]
    potential.sort()
    remaining_first, remaining_second = set(first), set(second)
    matches = []
    for _, a, b in potential:
        if a in remaining_first and b in remaining_second:
            remaining_first.remove(a)
            remaining_second.remove(b)
            matches.append((a, b))
    matches.sort()
    return matches


def quat_to_rotation(q) -> np.ndarray:
    """Unit-normalized quaternion (qx, qy, qz, qw) -> (3,3) rotation matrix."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
