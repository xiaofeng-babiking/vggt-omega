"""ARKitScenes (DUSt3R-style preprocessed) vendor, against :class:`BaseSequence`.

iPad ``vga_wide`` RGB + upsampled LiDAR depth + ARKit poses. One scene::

    {data_root}/{split}/<scene_id>/
        vga_wide/<scene_id>_<ts>.jpg       RGB (640x480 or 480x640)
        lowres_depth/<scene_id>_<ts>.png   uint16 depth (mm), same HxW as RGB
        scene_metadata.npz                 images (N,), trajectories (N,4,4),
                                           intrinsics (N,6)=[w,h,fx,fy,cx,cy]

Conventions (anchored to the original ``ArkitScenesDataset`` loader):

* Depth: uint16 PNG millimetres (metres = value / 1000); 0 = invalid. No sky.
* Pose: ``trajectories`` are camera-to-world (OpenCV axes); :meth:`get_pose`
  returns c2w directly.
* Intrinsics: **per-frame** rows ``[w,h,fx,fy,cx,cy]``; :meth:`get_intrinsic`
  returns frame 0's K. Override via ``intrinsics=(fx, fy, cx, cy)``.
* metadata ``images`` entries end in ``.png`` (depth filename); RGB is the same
  stem with ``.jpg`` under ``vga_wide/``. Timestamp = filename ``_<ts>`` suffix.
  Frames are sorted by timestamp.

A :class:`BaseSequence` is one scene; ``seq_id`` is the scene id. ``split``
selects ``Training`` / ``Test``.
"""
from __future__ import annotations

import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class ArkitScenesSequence(BaseSequence):
    """One ARKitScenes scene as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0
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

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        split: str = "Training",
        depth_scale: float = 1000.0,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        self.split = split
        self.seq_dir = os.path.join(data_root, split, seq_id)
        if not os.path.isdir(self.seq_dir):  # accept a root pointing at the scene parent
            self.seq_dir = os.path.join(data_root, seq_id)
        self.depth_scale = float(depth_scale)
        self._intrinsics_override = intrinsics

        # Parallel arrays, sorted by timestamp.
        self._rgb_paths: List[str] = []
        self._depth_paths: List[str] = []
        self._timestamps: List[float] = []
        self._traj: Optional[np.ndarray] = None      # (N,4,4) c2w
        self._intr_rows: Optional[np.ndarray] = None  # (N,6)
        self._intrinsic: Optional[np.ndarray] = None

        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    @staticmethod
    def _parse_ts(name: str) -> float:
        stem = os.path.splitext(name)[0]
        return float(stem.split("_", 1)[1])

    def load_manifest(self) -> None:
        md = np.load(os.path.join(self.seq_dir, "scene_metadata.npz"), allow_pickle=True)
        images = [str(x) for x in md["images"]]
        traj = np.asarray(md["trajectories"], dtype=np.float64)
        intr = np.asarray(md["intrinsics"], dtype=np.float64)
        order = np.argsort([self._parse_ts(n) for n in images])
        for j in order:
            name = images[j]  # "<scene>_<ts>.png" (depth filename)
            stem = os.path.splitext(name)[0]
            self._rgb_paths.append(os.path.join(self.seq_dir, "vga_wide", stem + ".jpg"))
            self._depth_paths.append(os.path.join(self.seq_dir, "lowres_depth", stem + ".png"))
            self._timestamps.append(self._parse_ts(name))
        self._traj = traj[order]
        self._intr_rows = intr[order]
        if len(self._rgb_paths) == 0:
            raise ValueError(f"ARKitScenes {self.seq_id}: no frames")

    def _row_to_K(self, row: np.ndarray) -> np.ndarray:
        if row.size != 6:
            raise ValueError(f"ARKitScenes intrinsics row must be [w,h,fx,fy,cx,cy], got {row.size}")
        _, _, fx, fy, cx, cy = row.tolist()
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
            self._intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            assert self._intr_rows is not None
            self._intrinsic = self._row_to_K(self._intr_rows[0])

    def load_extrinsics(self) -> None:
        return None

    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._rgb_paths)

    def get_timestamp(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> float:
        return float(self._timestamps[int(frame_id)])

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._rgb_paths[int(frame_id)]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("ARKitScenes provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("ARKitScenes provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """16-bit mm depth PNG -> ``(H, W)`` float32 metres (0 -> 0)."""
        with Image.open(self._depth_paths[int(frame_id)]) as im:
            arr = np.asarray(im).astype(np.float32)
        depth = arr / self.depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("ARKitScenes provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (OpenCV axes)."""
        assert self._traj is not None
        c2w = self._traj[int(frame_id)]
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise ValueError(f"ARKitScenes {self.seq_id}: pose {frame_id} malformed/non-finite")
        return NumpySE3Pose.from_rot_mat(c2w[:3, :3], c2w[:3, 3])

    def read_pose_file(self, pose_file: str) -> np.ndarray:
        raise NotImplementedError("poses are not stored as per-frame files")

    def get_poses_cache_file(self, sensor_id: Union[int, str]) -> str:
        return ""  # single-file / computed source: no per-frame combine cache

    def get_poses(self, sensor_id: Union[int, str]) -> List[BaseSE3Pose]:
        return [self.get_pose(sensor_id, i) for i in range(self.get_length(sensor_id))]

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("ARKitScenes provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("ARKitScenes provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._rgb_paths[int(frame_id)]

    def __repr__(self) -> str:
        return f"ArkitScenesSequence(seq_id={self.seq_id!r}, frames={len(self._rgb_paths)})"
