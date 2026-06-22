"""NYU Depth V2 (Eigen test split) vendor, implemented against :class:`BaseSequence`.

The 654-frame NYU Depth V2 test set (Eigen split), pre-decoded::

    {NYU_DIR}/val/nyu_images/XXXXX.png   RGB 640x480 (654 frames)
    {NYU_DIR}/val/nyu_depths/XXXXX.npy   float32 (480,640) depth, paired by basename

Conventions (anchored to the original ``NyuDataset`` loader):

* Frames are **independent single RGB-D captures** with no temporal structure or
  pose GT. They are exposed here as one sensor's frame timeline purely for
  addressing; ``get_pose`` is identity and EXTRINSIC / INTRINSIC are NOT
  advertised (no on-disk pose; intrinsics are literature values).
* Depth: ``np.load`` -> float32 metres directly (scale 1.0); non-finite / negative
  -> 0. Dense filled Kinect depth (no invalid pixels in practice).
* Basenames are the non-contiguous original h5 ids; frames are enumerated by
  sorted glob, paired with depth by basename.

The standard published NYUv2 Kinect calibration is available via
:meth:`nyu_intrinsics` for unprojection, but INTRINSIC is not advertised as GT.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class NyuSequence(BaseSequence):
    """NYU Depth V2 test split as a :class:`BaseSequence` (independent RGB-D frames)."""

    SENSOR: int = 0

    # Standard published NYUv2 Kinect pinhole (fx, fy, cx, cy) at 640x480 (literature
    # values, not on-disk GT) -- exposed for unprojection but INTRINSIC not advertised.
    _NYU_INTRINSICS = (518.857901, 519.469611, 325.582245, 253.736166)
    # Standard Eigen evaluation crop of the 480x640 frame: rows [45,471), cols [41,601).
    EIGEN_CROP = (45, 471, 41, 601)

    # RGB + dense metric depth only. No poses, no on-disk intrinsics, no timestamps.
    _MODALITIES = frozenset({Modality.RGB, Modality.DEPTH})

    def __init__(self, data_root: str, seq_id: str = "val"):
        """Open the NYU test split.

        Args:
            data_root: NYU root containing ``<seq_id>/nyu_images`` and
                ``<seq_id>/nyu_depths``.
            seq_id: split sub-directory (default ``"val"`` — the only split on disk).
        """
        self.data_root = data_root
        self.seq_id = seq_id
        self._images_dir = os.path.join(data_root, seq_id, "nyu_images")
        self._depths_dir = os.path.join(data_root, seq_id, "nyu_depths")

        # Each frame: (name, rgb_path, depth_path).
        self._frames: List[Tuple[str, str, str]] = []
        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    # -- lifecycle ----------------------------------------------------------- #
    def load_manifest(self) -> None:
        have_depth = {
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(self._depths_dir, "*.npy"))
        }
        frames = []
        for rgb_path in sorted(glob.glob(os.path.join(self._images_dir, "*.png"))):
            name = os.path.splitext(os.path.basename(rgb_path))[0]
            if name not in have_depth:
                continue
            frames.append((name, rgb_path, os.path.join(self._depths_dir, name + ".npy")))
        if not frames:
            raise ValueError(f"NYU: no paired frames under {self.data_root}/{self.seq_id}")
        self._frames = frames

    def load_intrinsics(self) -> None:
        return None  # not advertised; nyu_intrinsics() available on demand

    def load_extrinsics(self) -> None:
        return None  # no poses

    @classmethod
    def nyu_intrinsics(cls) -> np.ndarray:
        """(3,3) literature NYUv2 Kinect K (for unprojection; not advertised GT)."""
        fx, fy, cx, cy = cls._NYU_INTRINSICS
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    # -- discovery ----------------------------------------------------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frames)

    def get_timestamp(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> float:
        raise NotImplementedError("NYU frames are independent captures with no timestamps")

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][1]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("NYU provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("NYU provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """``.npy`` depth -> ``(H, W)`` float32 metres (scale 1.0; neg/NaN -> 0)."""
        depth = np.load(self._frames[int(frame_id)][2]).astype(np.float32)
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("NYU provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Identity camera-to-world: NYU frames have no pose GT (each frame's world
        frame is defined to be its own camera frame)."""
        return NumpySE3Pose.identity(backend="numpy")

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        # Not advertised; literature K available via nyu_intrinsics().
        raise NotImplementedError("NYU has no on-disk intrinsics; use NyuSequence.nyu_intrinsics()")

    # -- per-sequence products ----------------------------------------------- #
    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("NYU provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("NYU provides no ground-truth point cloud")

    # -- manifest-backed path lookup ----------------------------------------- #
    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][1]

    # parse() is inherited from BaseSequence (concrete template over the getters).

    def __repr__(self) -> str:
        return f"NyuSequence(seq_id={self.seq_id!r}, frames={len(self._frames)})"
