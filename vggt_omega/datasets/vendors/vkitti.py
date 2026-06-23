"""Virtual KITTI 2 (preprocessed) vendor, implemented against :class:`BaseSequence`.

Flattened per-frame npz copy::

    {data_root}/train/Scene{NN}/<variation>/Camera_{0,1}/
        {idx:05d}_rgb.jpg     RGB 1242x375
        {idx:05d}_depth.png   uint16 single-channel, depth in CENTIMETRES
        {idx:05d}_cam.npz     'camera_pose' (4,4), 'camera_intrinsics' (3,3)

Conventions (anchored to the original ``VkittiDataset`` loader):

* Depth: 16-bit PNG centimetres (metres = value / 100). 65535 (655.35 m) is the
  sky/far clamp -> -1.0 (repo sky convention); non-finite -> 0 (invalid).
* Pose: ``camera_pose`` is camera-to-world (OpenCV axes); :meth:`get_pose`
  returns c2w directly. Metric scale (KITTI baseline recovered to 0.533 m).
* Intrinsics: per-frame ``camera_intrinsics`` K (globally constant fx=fy=725.0087,
  cx=620.5, cy=187.0); override via ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``Scene/variation/Camera_N`` stream; ``seq_id`` is
that relative path. ~10 Hz video but no on-disk timestamps -> TIMESTAMP absent.
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


class VkittiSequence(BaseSequence):
    """One Virtual KITTI 2 camera stream as a :class:`BaseSequence`."""

    SENSOR: int = 0
    _SKY_RAW = 65535.0
    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        depth_scale: float = 100.0,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        self.data_root = data_root
        self.seq_id = seq_id
        root = os.path.join(data_root, "train")
        if not os.path.isdir(os.path.join(root, seq_id)):
            root = data_root
        self.seq_dir = os.path.join(root, seq_id)
        self.depth_scale = float(depth_scale)
        self._intrinsics_override = intrinsics
        self._frames: List[Tuple[str, str, str, int]] = []
        self._poses: Optional[List[BaseSE3Pose]] = None
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.seq_dir, "*_rgb.jpg")):
            m = re.fullmatch(r"(\d+)_rgb\.jpg", os.path.basename(rgb_path))
            if m is None:
                continue
            stem = m.group(1)
            frames.append(
                (
                    rgb_path,
                    os.path.join(self.seq_dir, stem + "_depth.png"),
                    os.path.join(self.seq_dir, stem + "_cam.npz"),
                    int(stem),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"VKITTI {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with np.load(path) as cam:
            if "camera_pose" not in cam or "camera_intrinsics" not in cam:
                raise ValueError(f"VKITTI cam {path!r}: expected camera_pose/camera_intrinsics")
            return np.asarray(cam["camera_intrinsics"], dtype=np.float32), np.asarray(cam["camera_pose"], dtype=np.float64)

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
        raise NotImplementedError("VKITTI has no on-disk timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_rgb_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("VKITTI provides no semantic masks (segmentation not exposed)")

    def get_rgb_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("VKITTI provides no dynamic masks")

    def get_rgb_valid_mask(self, sensor_id, frame_id) -> np.ndarray:
        """No per-frame valid annotations: all-ones mask (True everywhere),
        shape == RGB (H, W). Safe for elementwise multiply."""
        h, w = self.get_rgb(sensor_id, frame_id).shape[:2]
        return np.ones((h, w), dtype=bool)


    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """16-bit cm depth PNG -> float32 metres; sky (65535) -> -1.0, invalid -> 0."""
        with Image.open(self._frames[int(frame_id)][1]) as im:
            arr = np.asarray(im).astype(np.float32)
        sky = arr >= self._SKY_RAW
        depth = arr / self.depth_scale
        depth[~np.isfinite(depth)] = 0.0
        depth[sky] = -1.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("VKITTI provides no depth confidence")


    def read_pose_file(self, pose_file: str) -> np.ndarray:
        with np.load(pose_file) as cam:
            return np.asarray(cam["camera_pose"], dtype=np.float64).reshape(4, 4)

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
        raise NotImplementedError("VKITTI provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("VKITTI provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"VkittiSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
