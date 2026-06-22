"""TartanAir vendor, implemented against :class:`BaseSequence`.

TartanAir is a synthetic (AirSim) photorealistic SLAM dataset; depth and poses
are exact simulator ground truth. The preprocessed flattened dump is::

    <data_root>/train/<env>/<Easy|Hard>/P0XX/
        NNNNNN_rgb.png      RGB uint8 640x480
        NNNNNN_depth.npy    float32 (480,640) metric metres (scale 1.0)
        NNNNNN_cam.npz      'camera_pose' (4,4) camera-to-world (OpenCV axes),
                            'camera_intrinsics' (3,3) pinhole K

Conventions (anchored to the original ``TartanAirDataset`` loader):

* Pose: ``camera_pose`` is camera-to-world ALREADY in the OpenCV optical frame
  (the NED remap was applied during preprocessing). :meth:`get_pose` returns it
  directly as c2w (no axis flip).
* Depth: exact simulator metres. Sky is encoded as very large finite depth
  (> ``sky_threshold``) -> mapped to -1.0 (repo sky convention); the ambiguous
  band ``(valid_max, sky_threshold]`` and non-finite/negative -> 0.0 (invalid).
* Intrinsics: globally constant fx = fy = 320, cx = 320, cy = 240 (640x480);
  override via ``intrinsics=(fx, fy, cx, cy)``. The per-frame ``camera_intrinsics``
  merely duplicates this.

A :class:`BaseSequence` is one ``<env>/<Easy|Hard>/P0XX``; ``seq_id`` is that
relative path. No timestamps in the dump -> TIMESTAMP not provided.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class TartanAirSequence(BaseSequence):
    """One TartanAir ``env/Difficulty/P0XX`` sequence as a :class:`BaseSequence`."""

    SENSOR: int = 0
    _FX, _FY = 320.0, 320.0
    _PRINCIPAL_POINT = (320.0, 240.0)

    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        valid_max: float = 1000.0,
        sky_threshold: float = 10000.0,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        root = os.path.join(data_root, "train")
        if not os.path.isdir(os.path.join(root, seq_id)):
            root = data_root
        self.seq_dir = os.path.join(root, seq_id)
        self.valid_max = float(valid_max)
        self.sky_threshold = float(sky_threshold)
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
        for rgb_path in glob.glob(os.path.join(self.seq_dir, "*_rgb.png")):
            stem = os.path.basename(rgb_path)[: -len("_rgb.png")]
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
            raise ValueError(f"TartanAir {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
        else:
            fx, fy = self._FX, self._FY
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

    def get_timestamp(self, sensor_id, frame_id) -> float:
        raise NotImplementedError("TartanAir dump has no timestamps")

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("TartanAir provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("TartanAir provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """``.npy`` depth -> float32 metres; sky -> -1.0, invalid -> 0.0."""
        depth = np.array(np.load(self._frames[int(frame_id)][1]), dtype=np.float32, copy=True)
        sky = depth > self.sky_threshold  # +Inf included; NaN excluded
        invalid = ~np.isfinite(depth) | (depth < 0) | ((depth > self.valid_max) & ~sky)
        depth[invalid] = 0.0
        depth[sky] = -1.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("TartanAir provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (OpenCV optical frame)."""
        path = self._frames[int(frame_id)][2]
        with np.load(path) as cam:
            if "camera_pose" not in cam:
                raise ValueError(f"TartanAir cam {path!r}: missing 'camera_pose' (has {sorted(cam.keys())})")
            c2w = np.asarray(cam["camera_pose"], dtype=np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise ValueError(f"TartanAir {self.seq_id}: pose {frame_id} malformed/non-finite")
        return NumpySE3Pose.from_rot_mat(c2w[:3, :3], c2w[:3, 3])

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    # -- per-sequence products ----------------------------------------------- #
    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("TartanAir provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("TartanAir provides no ground-truth point cloud")

    # -- manifest-backed path lookup ----------------------------------------- #
    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    # parse() is inherited from BaseSequence (concrete template over the getters).

    def __repr__(self) -> str:
        return f"TartanAirSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
