"""MegaDepth vendor, implemented against :class:`BaseSequence`.

Processed MegaDepth copy (DUSt3R-style: EXR depth + safetensor cameras)::

    {data_root}/{SCENE}/{SUB}/{stem}.jpg          RGB JPEG
    {data_root}/{SCENE}/{SUB}/{stem}.exr          float32 (H,W) MVS depth (SfM scale)
    {data_root}/{SCENE}/{SUB}/{stem}.safetensor   'cam2world' (4,4), 'intrinsics' (3,3)

Each ``SCENE/SUB`` is one COLMAP sub-reconstruction (its own world frame/scale).

Conventions (anchored to the original ``MegaDepthDataset`` loader):

* Depth: single-channel EXR float32 in **SfM scale, not metric**; 0 = invalid
  (includes sky — outdoor photos, so no sky encoding/mask). Degenerate
  "ordinal" frames (fewer than ``min_depth_unique`` distinct positive values)
  are zeroed entirely.
* Pose: ``cam2world`` is camera-to-world (OpenCV axes); :meth:`get_pose`
  returns c2w directly.
* Intrinsics: per-frame ``intrinsics`` K (fx == fy, centred), varies per frame;
  :meth:`get_intrinsic` returns frame 0's K. Override via ``intrinsics=(...)``.

A :class:`BaseSequence` is one ``<SCENE>/<SUB>``; ``seq_id`` is that relative
path. Frames keyed by ``*.safetensor`` stem. No timestamps. Not metric.
"""
from __future__ import annotations

import glob
import os

# EXR decoding requires this env var BEFORE cv2 is first imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
from typing import List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from safetensors import safe_open

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class MegaDepthSequence(BaseSequence):
    """One MegaDepth sub-reconstruction as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0
    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        min_depth_unique: int = 5,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        self.seq_dir = os.path.join(data_root, seq_id)
        self.min_depth_unique = int(min_depth_unique)
        self._intrinsics_override = intrinsics
        # Each frame: (rgb_path, depth_path, cam_path, stem).
        self._frames: List[Tuple[str, str, str, str]] = []
        self._poses: Optional[List[BaseSE3Pose]] = None
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        frames = []
        for cam_path in glob.glob(os.path.join(self.seq_dir, "*.safetensor")):
            stem = os.path.basename(cam_path)[: -len(".safetensor")]
            frames.append(
                (
                    os.path.join(self.seq_dir, stem + ".jpg"),
                    os.path.join(self.seq_dir, stem + ".exr"),
                    cam_path,
                    stem,
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"MegaDepth {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with safe_open(path, framework="np") as f:
            c2w = np.asarray(f.get_tensor("cam2world"), dtype=np.float64)
            K = np.asarray(f.get_tensor("intrinsics"), dtype=np.float32)
        return K, c2w

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
        raise NotImplementedError("MegaDepth has no timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("MegaDepth provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("MegaDepth provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """EXR MVS depth -> float32 (SfM scale); 0 = invalid. Degenerate-ordinal
        frames (< ``min_depth_unique`` distinct positive values) zeroed."""
        path = self._frames[int(frame_id)][1]
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"MegaDepth: could not read depth {path}")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        if self.min_depth_unique > 0:
            positive = depth[depth > 0]
            if positive.size and np.unique(positive).size < self.min_depth_unique:
                depth[:] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("MegaDepth provides no depth confidence")


    def read_pose_file(self, pose_file: str) -> np.ndarray:
        with safe_open(pose_file, framework="np") as f:
            return np.asarray(f.get_tensor("cam2world"), dtype=np.float64).reshape(4, 4)

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
        raise NotImplementedError("MegaDepth provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("MegaDepth provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"MegaDepthSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
