"""MVS-Synth vendor, implemented against :class:`BaseSequence`.

Photorealistic GTA-V synthetic dataset (preprocessed copy)::

    {data_root}/train/{SEQ:0000..0119}/
        rgb/%04d.jpg     RGB JPEG 960x540
        depth/%04d.npy   float32 (540,960) metric metres; 0 = sky
        cam/%04d.npz     'intrinsics' (3,3) float32, 'pose' (4,4) float64 c2w

Conventions (anchored to the original ``MvsSynthDataset`` loader):

* Depth: metric metres (scale 1.0). Sky was stored as 0; 0 / negative /
  non-finite -> -1.0 (repo sky convention). Valid depth reaches ~7800 m.
* Pose: ``cam['pose']`` is camera-to-world (OpenCV axes); :meth:`get_pose`
  returns it as c2w. (The stored rotations carry ~1e-5 non-orthonormality.)
* Intrinsics: per-frame ``cam['intrinsics']`` K (constant-ish ~579 focal at
  960x540); :meth:`get_intrinsic` returns frame 0's K. Override via
  ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``{SEQ}`` dir; ``seq_id`` is that name. No
timestamps -> TIMESTAMP not provided.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class MvsSynthSequence(BaseSequence):
    """One MVS-Synth sequence as a :class:`BaseSequence` (single camera)."""

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
        root = os.path.join(data_root, "train")
        if not os.path.isdir(os.path.join(root, seq_id)):
            root = data_root
        self.seq_dir = os.path.join(root, seq_id)
        self._intrinsics_override = intrinsics

        # Each frame: (rgb_path, depth_path, cam_path, frame_idx).
        self._frames: List[Tuple[str, str, str, int]] = []
        self._intrinsic: Optional[np.ndarray] = None

        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    # -- lifecycle ----------------------------------------------------------- #
    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.seq_dir, "rgb", "*.jpg")):
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
            raise ValueError(f"MVS-Synth {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with np.load(path) as cam:
            try:
                K = np.asarray(cam["intrinsics"], dtype=np.float32)
                c2w = np.asarray(cam["pose"], dtype=np.float64)
            except KeyError as e:
                raise ValueError(f"MVS-Synth cam {path!r}: missing key {e}") from e
        if K.shape != (3, 3) or c2w.shape != (4, 4):
            raise ValueError(f"MVS-Synth cam {path!r}: bad shapes K{K.shape} pose{c2w.shape}")
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
        return None  # single sensor

    # -- discovery ----------------------------------------------------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id, frame_id) -> float:
        raise NotImplementedError("MVS-Synth has no timestamps")

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("MVS-Synth provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("MVS-Synth provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """``.npy`` depth -> float32 metres; sky/invalid (<=0 or non-finite) -> -1.0."""
        depth = np.asarray(np.load(self._frames[int(frame_id)][1]), dtype=np.float32).copy()
        if depth.ndim != 2:
            raise ValueError(f"MVS-Synth depth {frame_id}: expected 2-D, got {depth.shape}")
        depth[(depth <= 0) | ~np.isfinite(depth)] = -1.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("MVS-Synth provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (OpenCV axes)."""
        _, c2w = self._read_cam(int(frame_id))
        if not np.isfinite(c2w).all():
            raise ValueError(f"MVS-Synth {self.seq_id}: pose {frame_id} is non-finite")
        return NumpySE3Pose.from_rot_mat(c2w[:3, :3], c2w[:3, 3])

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    # -- per-sequence products ----------------------------------------------- #
    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("MVS-Synth provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("MVS-Synth provides no ground-truth point cloud")

    # -- manifest-backed path lookup ----------------------------------------- #
    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    # parse() is inherited from BaseSequence (concrete template over the getters).

    def __repr__(self) -> str:
        return f"MvsSynthSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
