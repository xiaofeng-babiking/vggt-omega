"""ScanNet (preprocessed ``scans_train`` copy) vendor, against :class:`BaseSequence`.

DUSt3R/CUT3R-style per-frame extraction (not the raw ``.sens`` release)::

    {SCANNET_DIR}/scans_train/sceneXXXX_XX/
        color/%05d.jpg                 RGB
        depth/%05d.png                 16-bit PNG, millimetres
        cam/%05d.npz                   keys 'intrinsics' (3,3), 'pose' (4,4) c2w

Conventions (anchored to the original ``ScannetDataset`` loader):

* Depth: 16-bit PNG millimetres (metres = value / 1000); 0 = invalid -> 0.
* Pose: ``cam/*.npz['pose']`` is **camera-to-world** in the OpenCV optical frame;
  returned directly by :meth:`get_pose`. (Tracking-failure frames are non-finite.)
* Intrinsics: ``cam/*.npz['intrinsics']`` is a per-frame (3,3) K, constant within
  a scene. :meth:`get_intrinsic` returns the first frame's K; override via
  ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` models ONE scene; ``seq_id`` is the scene dir name (e.g.
``"scene0000_00"``). No on-disk clock -> TIMESTAMP synthesized as
``frame_number / 30`` (numeric filename stem, so frame-id gaps stay gaps).
"""
from __future__ import annotations

import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class ScannetSequence(BaseSequence):
    """One ScanNet scene as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0
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
        scans_subdir: str = "scans_train",
        depth_scale: float = 1000.0,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        self.seq_dir = os.path.join(data_root, scans_subdir, seq_id)
        self.depth_scale = float(depth_scale)
        self._intrinsics_override = intrinsics

        # Each frame: (color_path, depth_path, cam_path, frame_num).
        self._frames: List[Tuple[str, str, str, int]] = []
        self._intrinsic: Optional[np.ndarray] = None
        self._poses: Optional[List[BaseSE3Pose]] = None  # lazily built/cached

        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    # -- lifecycle ----------------------------------------------------------- #
    def load_manifest(self) -> None:
        color_dir = os.path.join(self.seq_dir, "color")
        if not os.path.isdir(color_dir):
            raise ValueError(f"ScanNet {self.seq_id}: missing color/ dir under {self.seq_dir}")
        frames = []
        for fname in os.listdir(color_dir):
            stem, ext = os.path.splitext(fname)
            if ext.lower() != ".jpg" or not stem.isdigit():
                continue
            frames.append(
                (
                    os.path.join(color_dir, fname),
                    os.path.join(self.seq_dir, "depth", stem + ".png"),
                    os.path.join(self.seq_dir, "cam", stem + ".npz"),
                    int(stem),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """``cam/%05d.npz`` -> (K (3,3) float32, c2w (4,4) float64)."""
        path = self._frames[frame_id][2]
        with np.load(path) as cam:
            if "intrinsics" not in cam or "pose" not in cam:
                raise ValueError(
                    f"ScanNet cam file {path!r}: expected 'intrinsics' and 'pose', got {sorted(cam.files)}"
                )
            return np.asarray(cam["intrinsics"], dtype=np.float32), np.asarray(cam["pose"], dtype=np.float64)

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
            self._intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            K, _ = self._read_cam(0)  # per-frame K is constant within a scene
            if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
                raise ValueError(f"ScanNet {self.seq_id}: invalid intrinsics\n{K!r}")
            self._intrinsic = K

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

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("ScanNet provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("ScanNet provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """16-bit mm depth PNG -> ``(H, W)`` float32 metres (0 -> 0)."""
        with Image.open(self._frames[int(frame_id)][1]) as im:
            arr = np.asarray(im).astype(np.float32)
        depth = arr / self.depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("ScanNet provides no depth confidence")

    def read_pose_file(self, pose_file: str) -> np.ndarray:
        """Decode one ``cam/%05d.npz`` to a ``(4, 4)`` c2w matrix."""
        with np.load(pose_file) as cam:
            return np.asarray(cam["pose"], dtype=np.float64).reshape(4, 4)

    def get_poses_cache_file(self, sensor_id: Union[int, str]) -> str:
        return os.path.join(self.seq_dir, "poses_cache.npz")

    def get_poses(self, sensor_id: Union[int, str]) -> List[BaseSE3Pose]:
        """Frame-sorted c2w poses, combining the per-frame cam npz into one cache
        file on first call (then a single read thereafter)."""
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
                pass  # read-only data dir: still return the poses, just don't persist
        self._poses = [NumpySE3Pose.from_rot_mat(m[:3, :3], m[:3, 3]) for m in mats]
        return self._poses

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (OpenCV optical frame)."""
        return self.get_poses(sensor_id)[int(frame_id)]

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    # -- per-sequence products ----------------------------------------------- #
    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("ScanNet provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("ScanNet provides no ground-truth point cloud")

    # -- manifest-backed path lookup ----------------------------------------- #
    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    # parse() is inherited from BaseSequence (concrete template over the getters).

    def __repr__(self) -> str:
        return f"ScannetSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
