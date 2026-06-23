"""CO3Dv2 (DUSt3R-style preprocessed copy) vendor, against :class:`BaseSequence`.

Object-centric multi-view captures (poses pre-converted to OpenCV c2w)::

    {data_root}/{category}/{seq_id}/
        images/frame{i:06d}.jpg                RGB JPEG (short side ~384)
        images/frame{i:06d}.npz                camera_pose (4,4) c2w,
                                               camera_intrinsics (3,3),
                                               maximum_depth () float
        depths/frame{i:06d}.jpg.geometric.png  uint16 PNG, per-frame normalized

Conventions (anchored to the original ``Co3dDataset`` loader):

* Pose: ``camera_pose`` is camera-to-world (OpenCV axes); :meth:`get_pose`
  returns c2w directly.
* Depth: per-frame normalized uint16 -> ``depth = png / 65535 * maximum_depth``
  (the scalar from the sibling npz). 0 = invalid. **NOT metric** (per-sequence
  SfM-arbitrary units); no sky encoding.
* Intrinsics: per-frame ``camera_intrinsics`` K (fx == fy), varies within a
  sequence; :meth:`get_intrinsic` returns frame 0's K. Override via
  ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``<category>/<seq_id>``; ``seq_id`` is that
relative path. Frame indices are non-contiguous. No timestamps.
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


class Co3dSequence(BaseSequence):
    """One CO3Dv2 object sequence as a :class:`BaseSequence` (single camera)."""

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
        self.seq_dir = os.path.join(data_root, seq_id)
        self._intrinsics_override = intrinsics
        # Each frame: (rgb_path, depth_path, meta_path, frame_idx).
        self._frames: List[Tuple[str, str, str, int]] = []
        self._poses: Optional[List[BaseSE3Pose]] = None
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        frames = []
        for rgb_path in glob.glob(os.path.join(self.seq_dir, "images", "frame*.jpg")):
            m = re.fullmatch(r"frame(\d+)\.jpg", os.path.basename(rgb_path))
            if m is None:
                continue
            stem = os.path.splitext(os.path.basename(rgb_path))[0]  # "frame000123"
            frames.append(
                (
                    rgb_path,
                    os.path.join(self.seq_dir, "depths", f"{os.path.basename(rgb_path)}.geometric.png"),
                    os.path.join(self.seq_dir, "images", stem + ".npz"),
                    int(m.group(1)),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"CO3D {self.seq_id}: no frames under {self.seq_dir}")
        self._frames = frames

    def _read_meta(self, frame_id: int):
        path = self._frames[frame_id][2]
        with np.load(path) as md:
            if "camera_pose" not in md or "camera_intrinsics" not in md:
                raise ValueError(f"CO3D meta {path!r}: expected camera_pose/camera_intrinsics")
            K = np.asarray(md["camera_intrinsics"], dtype=np.float32)
            c2w = np.asarray(md["camera_pose"], dtype=np.float64)
            max_depth = float(md["maximum_depth"]) if "maximum_depth" in md else 1.0
        return K, c2w, max_depth

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
            self._intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            self._intrinsic, _, _ = self._read_meta(0)

    def load_extrinsics(self) -> None:
        return None

    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id, frame_id) -> float:
        raise NotImplementedError("CO3D has no per-frame timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("CO3D provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("CO3D provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """Per-frame normalized uint16 PNG -> float32 (png/65535 * maximum_depth);
        0 -> 0 (invalid). NOT metric (per-sequence SfM units)."""
        fid = int(frame_id)
        with Image.open(self._frames[fid][1]) as im:
            arr = np.asarray(im).astype(np.float32)
        _, _, max_depth = self._read_meta(fid)
        depth = arr / 65535.0 * max_depth
        depth[~np.isfinite(depth) | (arr == 0)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("CO3D provides no depth confidence")


    def read_pose_file(self, pose_file: str) -> np.ndarray:
        with np.load(pose_file) as cam:
            return np.asarray(cam["camera_pose"], dtype=np.float64).reshape(4, 4)

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
        raise NotImplementedError("CO3D provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("CO3D provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"Co3dSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
