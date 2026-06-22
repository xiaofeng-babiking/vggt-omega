"""Spring (CVPR'23 synthetic movie benchmark) vendor, against :class:`BaseSequence`.

Preprocessed Spring export::

    {data_root}/{split}/{SEQ}/
        rgb/%04d.png     8-bit RGB 960x540
        depth/%04d.npy   float32 (540,960) depth in METRES; 0 = invalid
        cam/%04d.npz     'intrinsics' (3,3), 'pose' (4,4) camera-to-world (OpenCV)

Conventions (anchored to the original ``SpringDataset`` loader):

* Pose: ``pose`` is camera-to-world (OpenCV axes); :meth:`get_pose` returns c2w.
* Depth: float32 npy **metres** (scale 1.0); 0 = invalid. Sky is a large finite
  scene-dependent depth (no reliable sentinel) -> left as-is (not remapped).
* Intrinsics: per-frame ``intrinsics`` K (constant within a sequence, varies
  strongly across); override via ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``{SEQ}``; ``seq_id`` is that name. Smooth video
but no on-disk timestamps -> TIMESTAMP not provided.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class SpringSequence(BaseSequence):
    """One Spring sequence as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0
    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        split: str = "train",
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        self.split = split
        self.seq_dir = os.path.join(data_root, split, seq_id)
        if not os.path.isdir(self.seq_dir):
            self.seq_dir = os.path.join(data_root, seq_id)
        self._intrinsics_override = intrinsics
        self._frames: List[Tuple[str, str, str, int]] = []
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.seq_dir, "rgb", "*.png")):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            frames.append(
                (
                    rgb_path,
                    os.path.join(self.seq_dir, "depth", stem + ".npy"),
                    os.path.join(self.seq_dir, "cam", stem + ".npz"),
                    int(stem),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"Spring {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with np.load(path) as cam:
            if "pose" not in cam or "intrinsics" not in cam:
                raise ValueError(f"Spring cam {path!r}: expected pose/intrinsics")
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
        raise NotImplementedError("Spring export has no timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Spring provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Spring provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """``.npy`` depth -> float32 metres; non-finite/negative -> 0 (invalid)."""
        depth = np.asarray(np.load(self._frames[int(frame_id)][1]), dtype=np.float32).copy()
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Spring provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (OpenCV axes)."""
        _, c2w = self._read_cam(int(frame_id))
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise ValueError(f"Spring {self.seq_id}: pose {frame_id} malformed/non-finite")
        return NumpySE3Pose.from_rot_mat(c2w[:3, :3], c2w[:3, 3])

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Spring provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("Spring provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"SpringSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
