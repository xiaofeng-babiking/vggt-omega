"""OmniObject3D vendor, implemented against :class:`BaseSequence`.

Object-centric Blender renders: each object sequence has 100 views (``r_0``..
``r_99``) under a ``train/`` split::

    {data_root}/train/<category>/<category>_<NNN>/
        rgb/r_<i>.png     RGB 800x800 (white background)
        depth/r_<i>.npy   float32 (800,800) z-depth, background = 0.0
        cam/r_<i>.npz     'intrinsics' (3,3), 'pose' (4,4) camera-to-world

Conventions (anchored to the original ``OmniObject3DDataset`` loader):

* Pose: ``cam['pose']`` is camera-to-world in OpenCV axes (Blender->OpenCV flip
  applied upstream); :meth:`get_pose` returns c2w directly (no flip).
* Depth: z-depth along the optical axis in **normalized scene units** (NOT
  metric). Background / non-finite / negative -> 0 (invalid); no sky concept.
* Intrinsics: per-frame K, globally constant fx = fy = 1111.1111, cx = cy = 400
  (800x800); override via ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one object ``<category>/<category>_<NNN>``; ``seq_id``
is that relative path. Views are sphere-sampled (not a trajectory) and there are
no timestamps -> TIMESTAMP not provided.
"""
from __future__ import annotations

import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class OmniObject3DSequence(BaseSequence):
    """One OmniObject3D object's multi-view set as a :class:`BaseSequence`."""

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
        self._frames: List[Tuple[str, str, str, int]] = []
        self._poses: Optional[List[BaseSE3Pose]] = None
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        frames = []
        rgb_dir = os.path.join(self.seq_dir, "rgb")
        for entry in os.scandir(rgb_dir):
            stem, ext = os.path.splitext(entry.name)  # "r_42", ".png"
            if ext != ".png" or not stem.startswith("r_"):
                continue
            frames.append(
                (
                    entry.path,
                    os.path.join(self.seq_dir, "depth", f"{stem}.npy"),
                    os.path.join(self.seq_dir, "cam", f"{stem}.npz"),
                    int(stem.split("_", 1)[1]),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"OmniObject3D {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_cam(self, frame_id: int) -> Tuple[np.ndarray, np.ndarray]:
        path = self._frames[frame_id][2]
        with np.load(path) as cam:
            if "pose" not in cam or "intrinsics" not in cam:
                raise ValueError(f"OmniObject3D cam {path!r}: expected 'pose' and 'intrinsics'")
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
        raise NotImplementedError("OmniObject3D views are not a timed trajectory")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("OmniObject3D provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("OmniObject3D provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """``.npy`` z-depth (scene units, not metric) -> float32; bg/invalid -> 0."""
        depth = np.asarray(np.load(self._frames[int(frame_id)][1]), dtype=np.float32).copy()
        if depth.ndim != 2:
            raise ValueError(f"OmniObject3D depth {frame_id}: expected 2-D, got {depth.shape}")
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("OmniObject3D provides no depth confidence")


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
        raise NotImplementedError("OmniObject3D provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("OmniObject3D provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"OmniObject3DSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
