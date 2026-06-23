"""Hypersim vendor, implemented against :class:`BaseSequence`.

Preprocessed PNG/npy/npz copy of Apple's photorealistic synthetic indoor
dataset::

    {data_root}/ai_XXX_XXX/cam_NN/
        FFFFFF_rgb.png    8-bit tonemapped RGB 1024x768
        FFFFFF_depth.npy  float32 (768,1024) planar z-depth metres; NaN = invalid
        FFFFFF_cam.npz    'pose' (4,4) camera-to-world (OpenCV), 'intrinsics' (3,3)

Conventions (anchored to the original ``HypersimDataset`` loader):

* Depth: planar z-depth in metres (already converted, not ray distance). NaN /
  non-finite -> 0 (invalid). No trustworthy sky -> no sky encoding.
* Pose: ``cam['pose']`` is camera-to-world (OpenCV axes); :meth:`get_pose`
  returns c2w directly.
* Intrinsics: per-frame ``cam['intrinsics']`` K (constant within a cam dir,
  fx=fy~883-887, principal point at image centre). Override via
  ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``ai_XXX_XXX/cam_NN`` trajectory; ``seq_id`` is
that relative path. Frame indices may be non-contiguous. No timestamps.
"""
from __future__ import annotations

import glob
import os
import re
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class HypersimSequence(BaseSequence):
    """One Hypersim ``ai_XXX_XXX/cam_NN`` trajectory as a :class:`BaseSequence`."""

    SENSOR: int = 0
    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        self.seq_dir = os.path.join(data_root, seq_id)
        self._intrinsics_override = intrinsics
        self._frames: List[Tuple[str, str, str, int]] = []
        self._poses: Optional[List[BaseSE3Pose]] = None
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.seq_dir, "*_rgb.png")):
            m = re.fullmatch(r"(\d+)_rgb\.png", os.path.basename(rgb_path))
            if m is None:
                continue
            stem = m.group(1)
            frames.append(
                (
                    rgb_path,
                    os.path.join(self.seq_dir, stem + "_depth.npy"),
                    os.path.join(self.seq_dir, stem + "_cam.npz"),
                    int(stem),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"Hypersim {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with np.load(path) as cam:
            if "pose" not in cam or "intrinsics" not in cam:
                raise ValueError(f"Hypersim cam {path!r}: expected 'pose' and 'intrinsics'")
            return np.asarray(cam["intrinsics"], dtype=np.float32), np.asarray(cam["pose"], dtype=np.float64)

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
            self._intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            self._intrinsic, _ = self._read_cam(0)

    def load_extrinsics(self) -> None:
        return None

    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id, frame_id) -> float:
        raise NotImplementedError("Hypersim has no timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Hypersim provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Hypersim provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """``.npy`` planar z-depth -> float32 metres; NaN/non-finite -> 0 (invalid)."""
        depth = np.asarray(np.load(self._frames[int(frame_id)][1]), dtype=np.float32).copy()
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Hypersim provides no depth confidence")


    def read_pose_file(self, pose_file: str) -> np.ndarray:
        with np.load(pose_file) as cam:
            return np.asarray(cam["pose"], dtype=np.float64).reshape(4, 4)

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
        raise NotImplementedError("Hypersim provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("Hypersim provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"HypersimSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
