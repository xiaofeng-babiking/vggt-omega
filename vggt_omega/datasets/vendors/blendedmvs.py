"""BlendedMVS vendor, implemented against :class:`BaseSequence`.

Object-centric multi-view-stereo dataset; per-scene per-frame dump::

    <data_root>/<scene_id>/
        NNNNNNNN.jpg          RGB 512x384
        NNNNNNNN.exr          float32 single-channel z-depth (SfM units)
        NNNNNNNN.safetensor   'R_cam2world' (3,3), 't_cam2world' (3,),
                              'intrinsics' (3,3)

Conventions (anchored to the original ``BlendedMvsDataset`` loader):

* Depth: z-depth in **SfM units** (not metric); 0 = invalid (no sky encoding).
* Pose: ``R_cam2world`` / ``t_cam2world`` are camera-to-world (OpenCV axes);
  :meth:`get_pose` returns c2w directly.
* Intrinsics: per-frame ``intrinsics`` K (constant within a scene), native
  512x384 pixels. Override via ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``<scene_id>``; ``seq_id`` is that id. Views are
an unordered wide-baseline collection. No timestamps. Not metric.
"""
from __future__ import annotations

import glob
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
from typing import List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from safetensors import safe_open

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class BlendedMvsSequence(BaseSequence):
    """One BlendedMVS scene as a :class:`BaseSequence` (single camera)."""

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
        # Each frame: (rgb_path, depth_path, cam_path, stem).
        self._frames: List[Tuple[str, str, str, str]] = []
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
            raise ValueError(f"BlendedMVS {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with safe_open(path, framework="np") as f:
            R = np.asarray(f.get_tensor("R_cam2world"), dtype=np.float64)
            t = np.asarray(f.get_tensor("t_cam2world"), dtype=np.float64).reshape(3)
            K = np.asarray(f.get_tensor("intrinsics"), dtype=np.float32)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
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
        raise NotImplementedError("BlendedMVS views are unordered; no timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("BlendedMVS provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("BlendedMVS provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """EXR z-depth -> float32 (SfM scale); 0/invalid -> 0."""
        path = self._frames[int(frame_id)][1]
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"BlendedMVS: could not read depth {path}")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("BlendedMVS provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-view **camera-to-world** SE(3) pose (OpenCV axes)."""
        _, c2w = self._read_cam(int(frame_id))
        if not np.isfinite(c2w).all():
            raise ValueError(f"BlendedMVS {self.seq_id}: pose {frame_id} is non-finite")
        return NumpySE3Pose.from_rot_mat(c2w[:3, :3], c2w[:3, 3])

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("BlendedMVS provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("BlendedMVS provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"BlendedMvsSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
