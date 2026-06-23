"""Neural RGB-D (Azinovic et al., CVPR 2022) vendor, against :class:`BaseSequence`.

Synthetic indoor benchmark; one scene per sequence::

    <scene>/images/img{i}.png            RGB 640x480 8-bit (i NOT zero-padded)
    <scene>/depth/depth{i}.png           GT rendered depth, uint16 millimetres
    <scene>/depth_filtered/, depth_with_noise/   alternative depth variants
    <scene>/focal.txt                    single float focal (px, fx == fy)
    <scene>/poses.txt                    GT trajectory, 4x4 per frame (OpenGL axes)

Conventions (anchored to the original ``NeuralRgbdDataset`` loader):

* Depth: 16-bit PNG millimetres (metres = value / 1000); 0 = invalid -> 0.
* Pose: ``poses.txt`` holds camera-to-world with **OpenGL** camera axes (camera
  looks down -Z, +Y up). :meth:`get_pose` converts to the OpenCV optical frame
  (negate the Y/Z rotation columns: ``c2w_cv = c2w_gl @ diag(1,-1,-1)``) and
  returns the **camera-to-world** OpenCV pose. The GT ``poses.txt`` is NaN-free;
  the noisy ``trainval_poses.txt`` is ignored.
* Intrinsics: ``focal.txt`` gives fx == fy; principal point is ((W-1)/2,(H-1)/2)
  of the native frame. Override via ``intrinsics=(fx, fy, cx, cy)``.

Filenames are not zero-padded and indices are contiguous 0..N-1, aligned
positionally with the pose rows. No on-disk timestamps -> TIMESTAMP not provided.
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


class NeuralRgbdSequence(BaseSequence):
    """One Neural RGB-D scene as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0
    # OpenGL (x-right, y-up, z-back) -> OpenCV (x-right, y-down, z-forward).
    _GL_TO_CV_FLIP = np.diag([1.0, -1.0, -1.0])
    _DEPTH_VARIANTS = ("depth", "depth_filtered", "depth_with_noise")

    _MODALITIES = frozenset(
        {
            Modality.RGB,
            Modality.DEPTH,
            Modality.POSE,
            Modality.INTRINSIC,
            Modality.EXTRINSIC,
        }
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        depth_scale: float = 1000.0,
        depth_variant: str = "depth",
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        if depth_variant not in self._DEPTH_VARIANTS:
            raise ValueError(f"depth_variant must be one of {self._DEPTH_VARIANTS}, got {depth_variant!r}")
        self.data_root = data_root
        self.seq_id = seq_id
        self.seq_dir = os.path.join(data_root, seq_id)
        self.depth_scale = float(depth_scale)
        self.depth_variant = depth_variant
        self._intrinsics_override = intrinsics

        # Each frame: (rgb_path, depth_path, frame_idx).
        self._frames: List[Tuple[str, str, int]] = []
        self._focal: Optional[float] = None
        self._intrinsic: Optional[np.ndarray] = None
        self._c2w_opencv: Optional[np.ndarray] = None  # (N,4,4) lazily parsed

        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    # -- lifecycle ----------------------------------------------------------- #
    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.seq_dir, "images", "img*.png")):
            m = re.fullmatch(r"img(\d+)\.png", os.path.basename(rgb_path))
            if m is None:
                continue
            idx = int(m.group(1))
            frames.append(
                (rgb_path, os.path.join(self.seq_dir, self.depth_variant, f"depth{idx}.png"), idx)
            )
        frames.sort(key=lambda fr: fr[2])
        self._frames = frames

    def load_intrinsics(self) -> None:
        with open(os.path.join(self.seq_dir, "focal.txt")) as f:
            self._focal = float(f.read())
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
        else:
            h, w = self.read_image_size(self._frames[0][0])
            fx = fy = self._focal
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        self._intrinsic = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )

    def load_extrinsics(self) -> None:
        """Parse GT ``poses.txt`` (OpenGL c2w) -> cached OpenCV c2w ``(N,4,4)``."""
        path = os.path.join(self.seq_dir, "poses.txt")
        with open(path) as f:
            txt = f.read().replace("-nan(ind)", "nan").replace("nan(ind)", "nan")
        vals = np.asarray(txt.split(), dtype=np.float64)
        if vals.size == 0 or vals.size % 16 != 0:
            raise ValueError(f"Neural RGB-D poses {path!r}: expected multiple of 16, got {vals.size}")
        c2w_gl = vals.reshape(-1, 4, 4)
        if len(c2w_gl) != len(self._frames):
            raise ValueError(
                f"Neural RGB-D {self.seq_id}: {len(c2w_gl)} poses but {len(self._frames)} frames"
            )
        c2w_cv = c2w_gl.copy()
        c2w_cv[:, :3, :3] = c2w_gl[:, :3, :3] @ self._GL_TO_CV_FLIP  # OpenGL -> OpenCV axes
        self._c2w_opencv = c2w_cv

    # -- discovery ----------------------------------------------------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> float:
        raise NotImplementedError("Neural RGB-D provides no timestamps")

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_rgb_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Neural RGB-D provides no semantic masks")

    def get_rgb_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Neural RGB-D provides no dynamic masks")

    def get_rgb_valid_mask(self, sensor_id, frame_id) -> np.ndarray:
        """No per-frame valid annotations: all-ones mask (True everywhere),
        shape == RGB (H, W). Safe for elementwise multiply."""
        h, w = self.get_rgb(sensor_id, frame_id).shape[:2]
        return np.ones((h, w), dtype=bool)


    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """16-bit mm depth PNG -> ``(H, W)`` float32 metres (0 -> 0)."""
        with Image.open(self._frames[int(frame_id)][1]) as im:
            arr = np.asarray(im).astype(np.float32)
        depth = arr / self.depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Neural RGB-D provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (converted to OpenCV axes)."""
        assert self._c2w_opencv is not None
        c2w = self._c2w_opencv[int(frame_id)]
        if not np.isfinite(c2w).all():
            raise ValueError(f"Neural RGB-D {self.seq_id}: pose {frame_id} is non-finite")
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

    # -- per-sequence products ----------------------------------------------- #
    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Neural RGB-D provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("Neural RGB-D provides no ground-truth point cloud")

    # -- manifest-backed path lookup ----------------------------------------- #
    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    # parse() is inherited from BaseSequence (concrete template over the getters).

    def __repr__(self) -> str:
        return f"NeuralRgbdSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
