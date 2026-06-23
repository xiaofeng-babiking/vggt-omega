"""WildRGB-D vendor, implemented against :class:`BaseSequence`.

Object-centric RGB-D phone captures. One scene's layout::

    {data_root}/{category}/scenes/scene_XXX/
        rgb/{id:05d}.jpg        RGB JPEG (portrait ~384-386 x 512-515)
        depth/{id:05d}.png      uint16 PNG, millimetres
        metadata/{id:05d}.npz   camera_intrinsics (3,3) f64, camera_pose (4,4) f64

Conventions (anchored to the original ``WildRgbdDataset`` loader):

* Depth: uint16 PNG millimetres (metres = value / 1000); 0 = invalid. No sky.
* Pose: ``camera_pose`` is camera-to-world (OpenCV axes); :meth:`get_pose`
  returns c2w directly. The first frame's pose is ~identity (poses are
  per-scene relative but metric scale).
* Intrinsics: per-frame ``camera_intrinsics`` K (zero skew, fx == fy), constant
  within a scene; :meth:`get_intrinsic` returns frame 0's K. Override via
  ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``<category>/scenes/scene_XXX``; ``seq_id`` is
that relative path. Frame ids are non-contiguous subsamples (sorted numerically
they remain a video). No timestamps -> TIMESTAMP not provided.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class WildRgbdSequence(BaseSequence):
    """One WildRGB-D scene as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0
    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        depth_scale: float = 1000.0,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        self.seq_dir = os.path.join(data_root, seq_id)
        self.depth_scale = float(depth_scale)
        self._intrinsics_override = intrinsics
        self._frames: List[Tuple[str, str, str, int]] = []
        self._poses: Optional[List[BaseSE3Pose]] = None
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.seq_dir, "rgb", "*.jpg")):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            frames.append(
                (
                    rgb_path,
                    os.path.join(self.seq_dir, "depth", stem + ".png"),
                    os.path.join(self.seq_dir, "metadata", stem + ".npz"),
                    int(stem),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"WildRGB-D {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_meta(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with np.load(path) as md:
            if "camera_intrinsics" not in md or "camera_pose" not in md:
                raise ValueError(f"WildRGB-D meta {path!r}: expected camera_intrinsics/camera_pose")
            return np.asarray(md["camera_intrinsics"], dtype=np.float32), np.asarray(md["camera_pose"], dtype=np.float64)

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
            self._intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            self._intrinsic, _ = self._read_meta(0)

    def load_extrinsics(self) -> None:
        return None

    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id, frame_id) -> float:
        raise NotImplementedError("WildRGB-D has no per-frame timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("WildRGB-D provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("WildRGB-D provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """16-bit mm depth PNG -> ``(H, W)`` float32 metres (0 -> 0)."""
        with Image.open(self._frames[int(frame_id)][1]) as im:
            arr = np.asarray(im).astype(np.float32)
        depth = arr / self.depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("WildRGB-D provides no depth confidence")


    def read_pose_file(self, pose_file: str) -> np.ndarray:
        with np.load(pose_file) as cam:
            return np.asarray(cam["camera_pose"], dtype=np.float64).reshape(4, 4)

    def get_poses_cache_file(self, sensor_id: Union[int, str]) -> str:
        return os.path.join(self.seq_dir, "poses_cache.npz")

    def get_poses(self, sensor_id: Union[int, str]) -> List[BaseSE3Pose]:
        if self._poses is not None:
            return self._poses
        cache = self.get_poses_cache_file(sensor_id)
        if os.path.exists(cache):
            with np.load(cache) as data:
                mats = np.asarray(data["poses"], dtype=np.float64)
        else:
            mats = np.stack([self.read_pose_file(fr[2]) for fr in self._frames], axis=0)
            try:
                np.savez(cache, poses=mats)
            except OSError:
                pass
        self._poses = [NumpySE3Pose.from_rot_mat(m[:3, :3], m[:3, 3]) for m in mats]
        return self._poses

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        return self.get_poses(sensor_id)[int(frame_id)]


    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("WildRGB-D provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("WildRGB-D provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"WildRgbdSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
