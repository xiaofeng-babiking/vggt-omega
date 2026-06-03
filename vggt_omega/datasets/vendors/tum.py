"""TUM RGB-D vendor for the VGGT-Omega dataset API.

TUM sequences provide RGB + depth (16-bit PNG, meters = pixel/5000) + a
groundtruth trajectory (timestamp tx ty tz qx qy qz qw, camera-to-world in the
color optical / OpenCV frame). RGB, depth and groundtruth are on different
clocks, so frames are associated by nearest timestamp. Intrinsics are per-camera
constants (freiburg1/2/3).
"""
from __future__ import annotations

import glob
import logging
import os

import numpy as np
from PIL import Image

# Official TUM per-camera pinhole intrinsics (fx, fy, cx, cy).
_TUM_INTRINSICS = {
    "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989),
    "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764),
    "freiburg3": (535.4, 539.2, 320.1, 247.6),
}


def read_file_list(path: str) -> dict[float, list[str]]:
    """Parse a TUM index file ('timestamp v1 v2 ...') -> {timestamp: [v1, v2, ...]}."""
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


def tum_pose_to_w2c(t, q) -> np.ndarray:
    """TUM (translation t, quaternion q) camera-to-world -> world-to-camera (3,4) OpenCV."""
    t = np.asarray(t, dtype=np.float64).reshape(3)
    rot_c2w = quat_to_rotation(q)              # camera-to-world rotation
    rot_w2c = rot_c2w.T
    trans_w2c = -rot_w2c @ t
    return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)


def read_tum_depth(path: str, depth_scale: float = 5000.0) -> np.ndarray:
    """Read a TUM 16-bit depth PNG -> float32 (H,W) meters. 0 stays 0 (invalid).

    NOTE: TUM stores plain uint16 integer counts (meters = value / depth_scale).
    This is NOT the float16-bit encoding the vendored dataset_util.read_depth
    assumes, so TUM uses this dedicated reader.
    """
    arr = np.asarray(Image.open(path)).astype(np.float32)
    depth = arr / float(depth_scale)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def tum_intrinsics(seq_name: str, override=None) -> np.ndarray:
    """(3,3) pinhole K for a TUM sequence. `override`=[fx,fy,cx,cy] wins; else the
    freiburg1/2/3 table is matched by substring. Raises ValueError if unknown."""
    if override is not None:
        fx, fy, cx, cy = override
    else:
        cam = next((k for k in _TUM_INTRINSICS if k in seq_name), None)
        if cam is None:
            raise ValueError(
                f"no TUM intrinsics for sequence {seq_name!r}; pass intrinsics=[fx,fy,cx,cy]"
            )
        fx, fy, cx, cy = _TUM_INTRINSICS[cam]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def associate_tum_sequence(seq_dir: str, max_diff: float = 0.02):
    """Associate rgb/depth/groundtruth in a TUM sequence dir.

    Returns a list of (rgb_path, depth_path, w2c (3,4) float32, timestamp float).
    """
    rgb = read_file_list(os.path.join(seq_dir, "rgb.txt"))
    depth = read_file_list(os.path.join(seq_dir, "depth.txt"))
    gt = read_file_list(os.path.join(seq_dir, "groundtruth.txt"))
    gt_ts = sorted(gt)
    frames = []
    for t_rgb, t_dep in associate(list(rgb), list(depth), max_diff):
        t_gt = min(gt_ts, key=lambda g: abs(g - t_rgb))
        if abs(t_gt - t_rgb) > max_diff:
            continue
        tx, ty, tz, qx, qy, qz, qw = (float(v) for v in gt[t_gt])
        w2c = tum_pose_to_w2c(np.array([tx, ty, tz]), (qx, qy, qz, qw))
        frames.append(
            (
                os.path.join(seq_dir, rgb[t_rgb][0]),
                os.path.join(seq_dir, depth[t_dep][0]),
                w2c,
                t_rgb,
            )
        )
    return frames
