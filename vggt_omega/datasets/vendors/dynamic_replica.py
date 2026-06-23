"""Dynamic Replica vendor, implemented against :class:`BaseSequence`.

Synthetic stereo 30 fps video of Replica scenes with animated objects::

    {data_root}/train/<6hex>-<N>_obj/{left,right}/
        rgb/<t>.png      RGBA uint8 1280x720 (alpha dropped)
        depth/<t>.npy    float32 (720,1280), ALREADY METRES, 0 = invalid
        cam/<t>.npz      'pose' (4,4) c2w, 'intrinsics' (3,3)

Each camera stream (``<seq>/left`` or ``<seq>/right``) is one sequence.

Conventions (anchored to the original ``DynamicReplicaDataset`` loader):

* Pose: ``pose`` is camera-to-world (OpenCV axes); :meth:`get_pose` returns c2w.
* Depth: float32 npy **metres** (scale 1.0); 0 / negative -> 0 (invalid). Indoor
  synthetic -> no sky.
* Intrinsics: per-frame ``intrinsics`` K (globally constant fx=fy=700, cx=640,
  cy=360 at 1280x720); override via ``intrinsics=(fx, fy, cx, cy)``.
* Frame filenames are float-second timestamps ``i/30``; frames are sorted by
  ``float(stem)`` and the stem doubles as the TIMESTAMP.

A :class:`BaseSequence` is one ``<seq>/{left,right}`` stream; ``seq_id`` is that
relative path.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class DynamicReplicaSequence(BaseSequence):
    """One Dynamic Replica camera stream as a :class:`BaseSequence`."""

    SENSOR: int = 0
    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.TIMESTAMP, Modality.INTRINSIC, Modality.EXTRINSIC}
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
        # Each frame: (rgb_path, depth_path, cam_path, timestamp, stem).
        self._frames: List[Tuple[str, str, str, float, str]] = []
        self._poses: Optional[List[BaseSE3Pose]] = None
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
                    float(stem),
                    stem,
                )
            )
        frames.sort(key=lambda fr: fr[3])  # by float timestamp, not lexical
        if not frames:
            raise ValueError(f"DynamicReplica {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with np.load(path) as cam:
            if "pose" not in cam or "intrinsics" not in cam:
                raise ValueError(f"DynamicReplica cam {path!r}: expected pose/intrinsics")
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

    def get_timestamp(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> float:
        return float(self._frames[int(frame_id)][3])

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """Decode RGBA PNG -> ``(H, W, 3)`` uint8 (alpha dropped)."""
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Dynamic Replica provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Dynamic Replica provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """``.npy`` depth -> float32 metres; non-finite/negative -> 0 (invalid)."""
        depth = np.asarray(np.load(self._frames[int(frame_id)][1]), dtype=np.float32).copy()
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Dynamic Replica provides no depth confidence")


    def read_pose_file(self, pose_file: str) -> np.ndarray:
        with np.load(pose_file) as cam:
            return np.asarray(cam["pose"], dtype=np.float64).reshape(4, 4)

    def get_poses_cache_file(self, sensor_id: Union[int, str]) -> str:
        return os.path.join(self.seq_dir, "poses_cache.npz")

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
        # Optical flow exists on disk but there is no FLOW modality / track GT here.
        raise NotImplementedError("Dynamic Replica flow is not exposed as tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("Dynamic Replica provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"DynamicReplicaSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
