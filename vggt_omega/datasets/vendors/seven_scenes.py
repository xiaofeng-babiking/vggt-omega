"""Microsoft 7-Scenes vendor, implemented against :class:`BaseSequence`.

7-Scenes is an indoor RGB-D relocalization benchmark (Kinect v1). Each scene
(chess, fire, heads, office, pumpkin, redkitchen, stairs) holds several
``seq-NN`` directories of ~1000 sequential frames at 30 Hz, four files per frame::

    frame-XXXXXX.color.png       RGB 640x480
    frame-XXXXXX.depth.png       raw Kinect depth (depth-sensor frame)
    frame-XXXXXX.depth.proj.png  depth registered into the color frame
    frame-XXXXXX.pose.txt        4x4 camera-to-world (OpenCV optical frame)

Conventions (anchored to the original ``SevenScenesDataset`` loader):

* Depth: 16-bit PNG millimetres (metres = value / 1000); 0 and 65535 are the
  invalid sentinels -> 0. ``.depth.proj.png`` (registered to color) is used by
  default; ``depth_variant="raw"`` selects the unregistered ``.depth.png``.
* Pose: ``pose.txt`` is camera-to-world in the OpenCV optical frame; returned
  directly by :meth:`get_pose`. (Failed-GT frames are marked non-finite.)
* Intrinsics: de-facto-standard focal 585 px, principal point (320, 240) at
  640x480; override via ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` models ONE ``seq-NN``; ``seq_id`` is ``"<scene>/seq-NN"``
(e.g. ``"chess/seq-01"``). No on-disk timestamps: synthesized as
``frame_index / 30`` for video ordering.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class SevenScenesSequence(BaseSequence):
    """One 7-Scenes ``seq-NN`` as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0
    _FOCAL = 585.0
    _PRINCIPAL_POINT = (320.0, 240.0)
    _FPS = 30.0

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
        depth_scale: float = 1000.0,
        depth_variant: str = "proj",
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        if depth_variant not in ("proj", "raw"):
            raise ValueError(f"depth_variant must be 'proj' or 'raw', got {depth_variant!r}")
        self.data_root = data_root
        self.seq_id = seq_id
        self.seq_dir = os.path.join(data_root, seq_id)
        self.depth_scale = float(depth_scale)
        self._depth_suffix = ".depth.proj.png" if depth_variant == "proj" else ".depth.png"
        self._intrinsics_override = intrinsics

        # Each frame: (color_path, depth_path, pose_path, frame_num).
        self._frames: List[Tuple[str, str, str, int]] = []
        self._intrinsic: Optional[np.ndarray] = None

        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    # -- lifecycle ----------------------------------------------------------- #
    def load_manifest(self) -> None:
        frames = []
        for color_path in glob.glob(os.path.join(self.seq_dir, "frame-*.color.png")):
            stem = os.path.basename(color_path).split(".")[0]  # "frame-000123"
            frame_num = int(stem.split("-")[1])
            frames.append(
                (
                    color_path,
                    color_path.replace(".color.png", self._depth_suffix),
                    color_path.replace(".color.png", ".pose.txt"),
                    frame_num,
                )
            )
        frames.sort(key=lambda fr: fr[3])
        self._frames = frames

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
        else:
            fx = fy = self._FOCAL
            cx, cy = self._PRINCIPAL_POINT
        self._intrinsic = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )

    def load_extrinsics(self) -> None:
        return None  # single sensor

    # -- discovery ----------------------------------------------------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> float:
        """Synthesized ``frame_number / 30`` (no on-disk clock)."""
        return float(self._frames[int(frame_id)][3]) / self._FPS

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_rgb_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("7-Scenes provides no semantic masks")

    def get_rgb_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("7-Scenes provides no dynamic masks")

    def get_rgb_valid_mask(self, sensor_id, frame_id) -> np.ndarray:
        """No per-frame valid annotations: all-ones mask (True everywhere),
        shape == RGB (H, W). Safe for elementwise multiply."""
        h, w = self.get_rgb(sensor_id, frame_id).shape[:2]
        return np.ones((h, w), dtype=bool)


    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """16-bit mm depth PNG -> ``(H, W)`` float32 metres (0/65535 -> 0)."""
        with Image.open(self._frames[int(frame_id)][1]) as im:
            arr = np.asarray(im).astype(np.float32)
        invalid = (arr == 0) | (arr == 65535) | ~np.isfinite(arr)
        depth = arr / self.depth_scale
        depth[invalid] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("7-Scenes provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (OpenCV optical frame)."""
        path = self._frames[int(frame_id)][2]
        with open(path) as f:
            vals = np.asarray(f.read().split(), dtype=np.float64)
        if vals.size != 16:
            raise ValueError(f"7-Scenes pose {path!r}: expected 16 values, got {vals.size}")
        c2w = vals.reshape(4, 4)
        if not np.isfinite(c2w).all():
            raise ValueError(f"7-Scenes pose {path!r} is non-finite (failed GT frame)")
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
        raise NotImplementedError("7-Scenes provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("7-Scenes provides no ground-truth point cloud")

    # -- manifest-backed path lookup ----------------------------------------- #
    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    # parse() is inherited from BaseSequence (concrete template over the getters).

    def __repr__(self) -> str:
        return f"SevenScenesSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
