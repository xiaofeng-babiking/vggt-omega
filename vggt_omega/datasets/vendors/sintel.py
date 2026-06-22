"""MPI Sintel vendor, implemented against :class:`BaseSequence`.

Synthetic film with dense GT depth and per-frame cameras (``training/`` only)::

    {data_root}/training/
        clean/<seq>/frame_%04d.png      RGB 1024x436 (effects-free pass)
        final/<seq>/frame_%04d.png      RGB (motion blur / fog pass)
        depth/<seq>/frame_%04d.dpt      float32 depth metres (PIEH binary)
        camdata_left/<seq>/frame_%04d.cam  K (3,3) + world->camera (3,4) (PIEH)

Conventions (anchored to the original ``SintelDataset`` loader):

* ``.dpt``/``.cam`` are little-endian PIEH-tagged binaries (float32 tag 202021.25).
* Pose: the ``.cam`` extrinsic is stored DIRECTLY as **world->camera** (OpenCV
  axes). :meth:`get_pose` returns the **camera-to-world** (its inverse) for
  cross-vendor consistency.
* Depth: float32 metres. Sky/far is large finite depth; pixels >=
  ``sky_threshold`` (default 1000 m) -> -1.0 (repo sky convention).
* Intrinsics: per-frame K from the ``.cam`` file (varies for zoom shots).
  Override via ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``<seq>``; ``seq_id`` is that name.
``render_pass`` selects the RGB pass ("final" default). No timestamps.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose

_TAG_FLOAT = 202021.25


class SintelSequence(BaseSequence):
    """One MPI Sintel sequence as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0
    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        render_pass: str = "final",
        sky_threshold: float = 1000.0,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        if render_pass not in ("clean", "final"):
            raise ValueError(f"render_pass must be 'clean' or 'final', got {render_pass!r}")
        self.data_root = data_root
        self.seq_id = seq_id
        self.render_pass = render_pass
        self.sky_threshold = float(sky_threshold)
        self._intrinsics_override = intrinsics
        training = os.path.join(data_root, "training")
        self._train_dir = training if os.path.isdir(training) else data_root
        # Each frame: (rgb_path, depth_path, cam_path, frame_idx).
        self._frames: List[Tuple[str, str, str, int]] = []
        self._intrinsic: Optional[np.ndarray] = None
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    def load_manifest(self) -> None:
        rgb_dir = os.path.join(self._train_dir, self.render_pass, self.seq_id)
        frames = []
        for rgb_path in glob.glob(os.path.join(rgb_dir, "frame_*.png")):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            idx = int(stem.split("_")[1])
            frames.append(
                (
                    rgb_path,
                    os.path.join(self._train_dir, "depth", self.seq_id, stem + ".dpt"),
                    os.path.join(self._train_dir, "camdata_left", self.seq_id, stem + ".cam"),
                    idx,
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if not frames:
            raise ValueError(f"Sintel {self.seq_id}: no frames in pass {self.render_pass!r}")
        self._frames = frames

    @staticmethod
    def _read_dpt(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            tag = np.fromfile(f, np.float32, 1)
            if tag.size != 1 or abs(float(tag[0]) - _TAG_FLOAT) > 1e-2:
                raise ValueError(f"Sintel depth {path!r}: bad PIEH tag {tag}")
            dims = np.fromfile(f, np.int32, 2)
            width, height = int(dims[0]), int(dims[1])
            data = np.fromfile(f, np.float32, width * height)
        if data.size != width * height:
            raise ValueError(f"Sintel depth {path!r}: truncated")
        return data.reshape(height, width)

    @staticmethod
    def _read_cam(path: str) -> Tuple[np.ndarray, np.ndarray]:
        with open(path, "rb") as f:
            tag = np.fromfile(f, np.float32, 1)
            if tag.size != 1 or abs(float(tag[0]) - _TAG_FLOAT) > 1e-2:
                raise ValueError(f"Sintel cam {path!r}: bad PIEH tag {tag}")
            K = np.fromfile(f, np.float64, 9).reshape(3, 3)
            w2c = np.fromfile(f, np.float64, 12).reshape(3, 4)
        return K.astype(np.float32), w2c

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
            self._intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            self._intrinsic, _ = self._read_cam(self._frames[0][2])

    def load_extrinsics(self) -> None:
        return None

    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id, frame_id) -> float:
        raise NotImplementedError("Sintel ships no per-frame timestamps")

    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Sintel provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Sintel provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """``.dpt`` depth -> float32 metres; sky (>= sky_threshold) -> -1.0."""
        depth = self._read_dpt(self._frames[int(frame_id)][1]).astype(np.float32).copy()
        sky = depth >= self.sky_threshold
        depth[~np.isfinite(depth)] = 0.0
        depth[sky] = -1.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Sintel provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (inverse of the stored w2c)."""
        _, w2c = self._read_cam(self._frames[int(frame_id)][2])
        if not np.isfinite(w2c).all():
            raise ValueError(f"Sintel {self.seq_id}: pose {frame_id} non-finite")
        return NumpySE3Pose.from_rot_mat(w2c[:3, :3], w2c[:3, 3]).inverse()

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Sintel provides no 2D/3D tracks (flow not exposed)")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("Sintel provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    def __repr__(self) -> str:
        return f"SintelSequence(seq_id={self.seq_id!r}, pass={self.render_pass!r}, frames={len(self._frames)})"
