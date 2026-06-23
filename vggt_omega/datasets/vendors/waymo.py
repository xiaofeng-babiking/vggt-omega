"""Waymo Open Dataset (preprocessed extraction) vendor, against :class:`BaseSequence`.

Each segment is a directory of preprocessed per-frame files (cv2 + numpy only)::

    {data_root}/train/segment-<id>_with_camera_labels.tfrecord/
        {frame:05d}_{cam}.jpg    RGB (cam in 1..5)
        {frame:05d}_{cam}.exr    sparse LiDAR depth (float32 metres)
        {frame:05d}_{cam}.npz    'intrinsics' (3,3), 'cam2world' (4,4), 'distortion' (5,)

Conventions (anchored to the original ``WaymoDataset`` loader):

* One sequence = one segment x one camera (a single ~10 Hz stream). ``camera``
  selects the Waymo camera id (1=FRONT, 2=FRONT_LEFT, 3=FRONT_RIGHT,
  4=SIDE_LEFT, 5=SIDE_RIGHT).
* Depth ``.exr`` is single-channel float32 **metres** (sparse LiDAR, ~11-17%
  valid); 0 = no return / invalid. No sky encoding.
* Pose: ``cam2world`` is camera-to-world (OpenCV axes); the world frame has huge
  offsets, so poses are **recentered per segment** (subtract the first frame's
  camera position). :meth:`get_pose` returns the recentered c2w.
* Intrinsics: per-frame ``intrinsics`` K; :meth:`get_intrinsic` returns frame
  0's K. Override via ``intrinsics=(fx, fy, cx, cy)``.
* 10 Hz video -> TIMESTAMP synthesized as ``frame_index / 10``.

A :class:`BaseSequence` is one ``<segment>`` + ``camera``; ``seq_id`` is the
segment directory name.
"""
from __future__ import annotations

import glob
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import re
from typing import List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class WaymoSequence(BaseSequence):
    """One Waymo segment x camera stream as a :class:`BaseSequence`."""

    SENSOR: int = 0
    _FPS = 10.0
    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.TIMESTAMP, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        camera: int = 1,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        self.camera = int(camera)
        root = os.path.join(data_root, "train")
        if not os.path.isdir(os.path.join(root, seq_id)):
            root = data_root
        self.seq_dir = os.path.join(root, seq_id)
        self._intrinsics_override = intrinsics
        # Each frame: (rgb_path, depth_path, cam_path, frame_idx).
        self._frames: List[Tuple[str, str, str, int]] = []
        self._poses: Optional[List[BaseSE3Pose]] = None
        self._origin: Optional[np.ndarray] = None  # per-segment recenter offset
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_extrinsics()
        self.load_intrinsics()

    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.seq_dir, f"*_{self.camera}.jpg")):
            m = re.fullmatch(rf"(\d+)_{self.camera}\.jpg", os.path.basename(rgb_path))
            if m is None:
                continue
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            frames.append(
                (
                    rgb_path,
                    os.path.join(self.seq_dir, stem + ".exr"),
                    os.path.join(self.seq_dir, stem + ".npz"),
                    int(m.group(1)),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"Waymo {self.seq_id} cam {self.camera}: no frames under {self.seq_dir}")
        self._frames = frames

    def _raw_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with np.load(path) as cam:
            if "cam2world" not in cam or "intrinsics" not in cam:
                raise ValueError(f"Waymo cam {path!r}: expected cam2world/intrinsics")
            return np.asarray(cam["intrinsics"], dtype=np.float32), np.asarray(cam["cam2world"], dtype=np.float64)

    def load_extrinsics(self) -> None:
        # Per-segment recenter offset = first frame's camera position.
        _, c2w0 = self._raw_cam(0)
        self._origin = c2w0[:3, 3].copy()

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
            self._intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            self._intrinsic, _ = self._raw_cam(0)

    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> float:
        return float(self._frames[int(frame_id)][3]) / self._FPS

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_rgb_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Waymo (this copy) provides no semantic masks")

    def get_rgb_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Waymo (this copy) provides no dynamic masks")

    def get_rgb_valid_mask(self, sensor_id, frame_id) -> np.ndarray:
        """No per-frame valid annotations: all-ones mask (True everywhere),
        shape == RGB (H, W). Safe for elementwise multiply."""
        h, w = self.get_rgb(sensor_id, frame_id).shape[:2]
        return np.ones((h, w), dtype=bool)


    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """EXR sparse LiDAR depth -> float32 metres; 0 = no return (invalid)."""
        path = self._frames[int(frame_id)][1]
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Waymo: could not read depth {path}")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Waymo provides no depth confidence")


    def read_pose_file(self, pose_file: str) -> np.ndarray:
        with np.load(pose_file) as cam:
            return np.asarray(cam["cam2world"], dtype=np.float64).reshape(4, 4)

    def get_poses_cache_file(self, sensor_id: Union[int, str]) -> str:
        # Data lives under a read-only mount; mirror the seq dir under a writable
        # root so poses_cache.npz can be written. Remap is overridable via
        # VGGT_POSE_CACHE_REMAP="<src_prefix>:<dst_prefix>".
        src, _, dst = os.environ.get(
            "VGGT_POSE_CACHE_REMAP", "/jfs/Data_4DFF:/jfs/jing.feng/Data_4DFF"
        ).partition(":")
        seq_dir = os.path.abspath(self.seq_dir)
        src, dst = os.path.abspath(src), os.path.abspath(dst)
        cache_dir = dst + seq_dir[len(src):] if seq_dir.startswith(src) else seq_dir
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError:
            return ""
        return os.path.join(cache_dir, "poses_cache.npz")

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
        raise NotImplementedError("Waymo provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("Waymo provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"WaymoSequence(seq_id={self.seq_id!r}, cam={self.camera}, frames={len(self._frames)})"
