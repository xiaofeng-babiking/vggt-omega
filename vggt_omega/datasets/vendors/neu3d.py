"""Neu3D (DyNeRF) vendor, implemented against :class:`BaseSequence`.

This cluster's Neu3D copy is image-only: a monocular video of one DyNeRF scene
from one camera, with pre-extracted PNG frames and nothing else::

    {data_root}/<scene>/<cam>/
        images/0000.png .. NNNN.png          8-bit RGB
        downsampled_2x/0000.png .. NNNN.png  exact 2x downsample

Conventions (anchored to the original ``Neu3dDataset`` loader):

* **IMAGE-ONLY**: no depth, pose or intrinsics on disk -> only RGB is
  advertised. ``get_pose`` returns identity (a single static camera), but
  POSE / EXTRINSIC / INTRINSIC are NOT advertised (no GT).
* ``image_variant`` selects ``"images"`` (full) or ``"downsampled_2x"``.

A :class:`BaseSequence` is one ``<scene>/<cam>`` stream; ``seq_id`` is that
relative path. Temporally ordered video, but no timestamps on disk.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class Neu3dSequence(BaseSequence):
    """One Neu3D camera stream as a :class:`BaseSequence` (image-only)."""

    SENSOR: int = 0
    _IMAGE_VARIANTS = ("images", "downsampled_2x")
    _MODALITIES = frozenset({Modality.RGB})

    def __init__(self, data_root: str, seq_id: str, *, image_variant: str = "images"):
        if image_variant not in self._IMAGE_VARIANTS:
            raise ValueError(f"image_variant must be one of {self._IMAGE_VARIANTS}, got {image_variant!r}")
        self.data_root = data_root
        self.seq_id = seq_id
        self.image_variant = image_variant
        self.frames_dir = os.path.join(data_root, seq_id, image_variant)
        self._frames: List[Tuple[str, int]] = []
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.frames_dir, "*.png")):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            if not stem.isdigit():
                continue
            frames.append((rgb_path, int(stem)))
        frames.sort(key=lambda fr: fr[1])
        if not frames:
            raise ValueError(f"Neu3D {self.seq_id}: no frames under {self.frames_dir}")
        self._frames = frames

    def load_intrinsics(self) -> None:
        return None  # none on disk

    def load_extrinsics(self) -> None:
        return None

    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id, frame_id) -> float:
        raise NotImplementedError("Neu3D ships no per-frame timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Neu3D provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Neu3D provides no dynamic masks")

    def get_depth(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Neu3D provides no depth")

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Neu3D provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Identity (static camera; no pose GT). POSE is not advertised."""
        return NumpySE3Pose.identity(backend="numpy")

    def read_pose_file(self, pose_file: str) -> np.ndarray:
        raise NotImplementedError("poses are not stored as per-frame files")

    def get_poses_cache_file(self, sensor_id: Union[int, str]) -> str:
        return ""  # single-file / computed source: no per-frame combine cache

    def get_poses(self, sensor_id: Union[int, str]) -> List[BaseSE3Pose]:
        return [self.get_pose(sensor_id, i) for i in range(self.get_length(sensor_id))]

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id) -> np.ndarray:
        raise NotImplementedError("Neu3D has no on-disk intrinsics")

    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Neu3D provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("Neu3D provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"Neu3dSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
