"""TUM RGB-D vendor, implemented against :class:`BaseSequence`.

A TUM RGB-D capture is a single moving RGB-D camera. The disk layout per
sequence directory is::

    <seq>/rgb.txt           # 'timestamp filename' index for the color frames
    <seq>/depth.txt         # 'timestamp filename' index for the depth frames
    <seq>/groundtruth.txt   # 'timestamp tx ty tz qx qy qz qw' camera-to-world
    <seq>/rgb/*.png         # 8-bit color
    <seq>/depth/*.png       # 16-bit depth, metres = pixel / 5000

RGB, depth and groundtruth run on three different clocks, so a frame is built by
greedy nearest-timestamp association (``vendors.common.associate``) keyed on the
color timestamps. Intrinsics are per-camera constants (freiburg1 / 2 / 3).

This class is a :class:`BaseSequence`: it models **one** sequence with **one**
sensor (``sensor_id == 0``, the RGB-D camera). Construction reads only the text
indices (no pixels); the ``get_rgb`` / ``get_depth`` getters decode lazily.
Poses are returned as :class:`NumpySE3Pose` value objects.
"""
from __future__ import annotations

import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose
from vggt_omega.datasets.vendors.common import (
    associate,
    quat_to_rotation,
    read_file_list,
)


class TumSequence(BaseSequence):
    """One TUM RGB-D sequence as a :class:`BaseSequence`.

    Single sensor (``SENSOR = 0``). Frames are RGB / depth pairs associated to a
    groundtruth pose by nearest timestamp; ``frame_id`` is the integer index into
    that associated, timestamp-sorted list.
    """

    SENSOR: int = 0  # TUM is a single RGB-D camera

    # Official TUM per-camera pinhole intrinsics (fx, fy, cx, cy).
    _TUM_INTRINSICS = {
        "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989),
        "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764),
        "freiburg3": (535.4, 539.2, 320.1, 247.6),
    }

    _MODALITIES = frozenset(
        {
            Modality.RGB,
            Modality.DEPTH,
            Modality.POSE,
            Modality.TIMESTAMP,
            Modality.INTRINSIC,
            Modality.EXTRINSIC,
        }
    )

    # -- lifecycle ----------------------------------------------------------- #
    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        assoc_max_diff: float = 0.02,
        depth_scale: float = 5000.0,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        """Open a TUM sequence directory.

        Args:
            data_root: directory containing the TUM sequences.
            seq_id: sequence sub-directory name (e.g.
                ``"rgbd_dataset_freiburg1_xyz"``).
            assoc_max_diff: max timestamp gap (s) for rgb/depth/gt association.
            depth_scale: TUM depth PNG scale (metres = pixel / depth_scale).
            intrinsics: optional ``(fx, fy, cx, cy)`` override; otherwise inferred
                from the freiburg1/2/3 substring in ``seq_id``.
        """
        self.data_root = data_root
        self.seq_id = seq_id
        self.seq_dir = os.path.join(data_root, seq_id)
        self.assoc_max_diff = float(assoc_max_diff)
        self.depth_scale = float(depth_scale)
        self._intrinsics_override = intrinsics

        # Per-frame records, built by load_manifest(): each is
        # (rgb_path, depth_path, timestamp, t_c2w (3,), quat_xyzw (4,)).
        self._frames: List[Tuple[str, str, float, np.ndarray, np.ndarray]] = []
        self._intrinsic: Optional[np.ndarray] = None

        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        """Parse rgb/depth/groundtruth indices and associate them by timestamp."""
        rgb = read_file_list(os.path.join(self.seq_dir, "rgb.txt"))
        depth = read_file_list(os.path.join(self.seq_dir, "depth.txt"))
        gt = read_file_list(os.path.join(self.seq_dir, "groundtruth.txt"))
        gt_ts = sorted(gt)

        frames: List[Tuple[str, str, float, np.ndarray, np.ndarray]] = []
        for t_rgb, t_dep in associate(list(rgb), list(depth), self.assoc_max_diff):
            t_gt = min(gt_ts, key=lambda g: abs(g - t_rgb))
            if abs(t_gt - t_rgb) > self.assoc_max_diff:
                continue
            tx, ty, tz, qx, qy, qz, qw = (float(v) for v in gt[t_gt])
            frames.append(
                (
                    os.path.join(self.seq_dir, rgb[t_rgb][0]),
                    os.path.join(self.seq_dir, depth[t_dep][0]),
                    float(t_rgb),
                    np.array([tx, ty, tz], dtype=np.float64),
                    np.array([qx, qy, qz, qw], dtype=np.float64),
                )
            )
        # associate() already returns matches sorted by t_rgb; keep that order.
        self._frames = frames

    def load_intrinsics(self) -> None:
        """Resolve the (constant) pinhole intrinsics for this sequence."""
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
        else:
            cam = next((k for k in self._TUM_INTRINSICS if k in self.seq_id), None)
            if cam is None:
                raise ValueError(
                    f"no TUM intrinsics for sequence {self.seq_id!r}; "
                    "pass intrinsics=(fx, fy, cx, cy)"
                )
            fx, fy, cx, cy = self._TUM_INTRINSICS[cam]
        self._intrinsic = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )

    def load_extrinsics(self) -> None:
        """No inter-sensor calibration: TUM has a single sensor (identity)."""
        # Nothing to load — get_extrinsic only accepts (SENSOR, SENSOR) -> identity.
        return None

    # -- discovery ----------------------------------------------------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> float:
        return float(self._frames[int(frame_id)][2])

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Decode the color frame -> ``(H, W, 3)`` uint8 RGB."""
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_rgb_semantic_mask(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no semantic masks")

    def get_rgb_dynamic_mask(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no dynamic masks")

    def get_rgb_valid_mask(self, sensor_id, frame_id) -> np.ndarray:
        """No per-frame valid annotations: all-ones mask (True everywhere),
        shape == RGB (H, W). Safe for elementwise multiply."""
        h, w = self.get_rgb(sensor_id, frame_id).shape[:2]
        return np.ones((h, w), dtype=bool)

    def get_depth(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Decode the 16-bit depth PNG -> ``(H, W)`` float32 metres (0 = invalid).

        TUM stores plain uint16 integer counts; metres = value / ``depth_scale``.
        """
        with Image.open(self._frames[int(frame_id)][1]) as im:
            arr = np.asarray(im).astype(np.float32)
        depth = arr / self.depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no depth confidence")

    def get_pose(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> BaseSE3Pose:
        """Per-frame camera-to-world SE(3) pose (TUM groundtruth convention)."""
        _, _, _, t_c2w, quat_xyzw = self._frames[int(frame_id)]
        return NumpySE3Pose.from_quat(quat_xyzw, t_c2w, scalar_last=True)

    def read_pose_file(self, pose_file: str) -> np.ndarray:
        raise NotImplementedError("poses are not stored as per-frame files")

    def get_poses_cache_file(self, sensor_id: Union[int, str]) -> str:
        return ""  # single-file / computed source: no per-frame combine cache

    def get_poses(self, sensor_id: Union[int, str]) -> List[BaseSE3Pose]:
        return [self.get_pose(sensor_id, i) for i in range(self.get_length(sensor_id))]

    def get_extrinsic(
        self, src_sensor_id: Union[int, str], dst_sensor_id: Union[int, str]
    ) -> BaseSE3Pose:
        """Static transform between sensors. TUM has one sensor -> identity."""
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        """``(3, 3)`` pinhole camera matrix K (float32)."""
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    # -- per-sensor / per-sequence products ---------------------------------- #
    def get_tracks(
        self, sensor_id: Union[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("TUM RGB-D provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        # TUM has no independent point-cloud GT (only depth re-projected through the
        # GT poses), so none is advertised here.
        raise NotImplementedError("TUM RGB-D provides no ground-truth point cloud")

    # -- manifest-backed path lookup (base reads image sizes through this) --- #
    def _frame_image_path(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> str:
        return self._frames[int(frame_id)][0]

    # parse() is inherited from BaseSequence (concrete template over the getters).

    @classmethod
    def discover(cls, data_root: str, patterns: Optional[List[str]] = None) -> List[str]:
        """TUM sequence ids = immediate sub-dirs of ``data_root`` that contain an
        ``rgb.txt`` index, filtered by the glob ``patterns`` (``["*"]`` if None)."""
        import glob

        names = set()
        for pat in (patterns or ["*"]):
            for d in glob.glob(os.path.join(data_root, pat)):
                if os.path.isfile(os.path.join(d, "rgb.txt")):
                    names.add(os.path.basename(d.rstrip("/")))
        return sorted(names)

    def __repr__(self) -> str:
        return f"TumSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
