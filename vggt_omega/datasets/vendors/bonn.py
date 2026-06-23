"""Bonn RGB-D Dynamic vendor, implemented against :class:`BaseSequence`.

Bonn RGB-D Dynamic (Palazzolo et al., IROS 2019) is a TUM-format indoor RGB-D
benchmark of mostly-dynamic sequences, nested one level below the dataset root::

    BONN_DIR/rgbd_bonn_dataset/rgbd_bonn_<scene>/
        rgb/<unix_ts>.png       RGB 640x480 uint8
        depth/<unix_ts>.png     16-bit depth registered to color, metres = value/5000
        rgb.txt, depth.txt      TUM index files ("timestamp path")
        groundtruth.txt         TUM trajectory "ts tx ty tz qx qy qz qw" (100 Hz mocap)
        rgb_110/, depth_110/,    pre-associated 110-frame eval subset (index-aligned),
        groundtruth_110.txt      groundtruth_110.txt is headerless numpy scientific text

Conventions (anchored to the original ``BonnDataset`` loader):

* Depth: uint16 PNG, metres = value / 5000 (TUM scale); 0 = invalid.
* Intrinsics: no calibration ships on disk; the official published Bonn
  calibration ``(542.822841, 542.576870, 315.593520, 237.756098)`` (640x480) is
  the default, overridable via ``intrinsics=(fx, fy, cx, cy)``.
* Pose frame: ``groundtruth.txt`` is camera-to-world of the OptiTrack **marker
  body**, not the color optical frame. ``pose_frame="camera"`` (default)
  right-multiplies the marker c2w by the constant hand-eye :attr:`MARKER_TO_CAM`
  (reproject-consistent OpenCV camera pose); ``pose_frame="marker"`` returns the
  raw published c2w (parity with published ATE protocols).
* ``subset="110"`` (default) uses the index-aligned 110-frame eval triplets;
  ``subset="full"`` associates the full rgb/depth/groundtruth streams by nearest
  timestamp (TUM-style, ``assoc_max_diff`` gate).

:meth:`get_pose` returns the **camera-to-world** pose (consistent with every
other vendor); ``parse`` emits the w2c extrinsic as its inverse.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose
from vggt_omega.datasets.vendors.common import associate, quat_to_rotation, read_file_list


class BonnSequence(BaseSequence):
    """One Bonn RGB-D Dynamic sequence as a :class:`BaseSequence` (single camera)."""

    SENSOR: int = 0

    # Official published Bonn pinhole calibration (fx, fy, cx, cy) for 640x480.
    _BONN_INTRINSICS = (542.822841, 542.576870, 315.593520, 237.756098)

    # Constant OptiTrack-marker-body -> color-camera (OpenCV) transform X, so that
    # c2w_camera = c2w_marker @ X (Tsai-Lenz hand-eye on this data).
    MARKER_TO_CAM = np.array(
        [
            [-0.96509344, -0.17127006, 0.19814445, 0.00130562],
            [-0.26129314, 0.57791660, -0.77313537, -0.00304165],
            [0.01790398, -0.79792166, -0.60249521, -0.01835103],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

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
        subset: str = "110",
        pose_frame: str = "camera",
        assoc_max_diff: float = 0.02,
        depth_scale: float = 5000.0,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        if subset not in ("110", "full"):
            raise ValueError(f"subset must be '110' or 'full', got {subset!r}")
        if pose_frame not in ("camera", "marker"):
            raise ValueError(f"pose_frame must be 'camera' or 'marker', got {pose_frame!r}")
        self.data_root = data_root
        self.seq_id = seq_id
        # The official archive nests sequences one level down; accept either.
        nested = os.path.join(data_root, "rgbd_bonn_dataset", seq_id)
        self.seq_dir = nested if os.path.isdir(nested) else os.path.join(data_root, seq_id)
        self.subset = subset
        self.pose_frame = pose_frame
        self.assoc_max_diff = float(assoc_max_diff)
        self.depth_scale = float(depth_scale)
        self._intrinsics_override = intrinsics

        # Each frame: (rgb_path, depth_path, timestamp, t_c2w (3,), quat_xyzw (4,)).
        self._frames: List[Tuple[str, str, float, np.ndarray, np.ndarray]] = []
        self._intrinsic: Optional[np.ndarray] = None

        self.load_manifest()
        self.load_intrinsics()
        self.load_extrinsics()

    # -- lifecycle ----------------------------------------------------------- #
    def _marker_c2w_to_camera_c2w(self, t: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Marker-body c2w (t, quat xyzw) -> camera c2w ``(t', quat' xyzw)``.

        Applies the hand-eye :attr:`MARKER_TO_CAM` when ``pose_frame=="camera"``;
        returns the raw marker pose otherwise. Returns ``(translation, quat_xyzw)``.
        """
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = quat_to_rotation(q)
        c2w[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
        if self.pose_frame == "camera":
            c2w = c2w @ self.MARKER_TO_CAM
        return c2w, NumpySE3Pose.from_rot_mat(c2w[:3, :3], c2w[:3, 3]).quaternion

    def _append_frame(self, rgb_path, depth_path, ts, t, q) -> None:
        c2w, quat_xyzw = self._marker_c2w_to_camera_c2w(t, q)
        self._frames.append(
            (rgb_path, depth_path, float(ts), c2w[:3, 3].copy(), quat_xyzw)
        )

    def load_manifest(self) -> None:
        if self.subset == "110":
            self._load_110()
        else:
            self._load_full()

    def _load_110(self) -> None:
        """Index-aligned 110-frame eval triplets (rgb_110/depth_110/groundtruth_110.txt)."""
        ts_key = lambda p: float(os.path.splitext(os.path.basename(p))[0])
        rgb_files = sorted(glob.glob(os.path.join(self.seq_dir, "rgb_110", "*.png")), key=ts_key)
        depth_files = sorted(glob.glob(os.path.join(self.seq_dir, "depth_110", "*.png")), key=ts_key)
        gt = np.atleast_2d(np.loadtxt(os.path.join(self.seq_dir, "groundtruth_110.txt")))
        if not (len(rgb_files) == len(depth_files) == len(gt)) or gt.shape[1] != 8:
            raise ValueError(
                f"Bonn {self.seq_id}: _110 subset not index-aligned "
                f"({len(rgb_files)} rgb, {len(depth_files)} depth, gt {gt.shape})"
            )
        for rgb_path, depth_path, row in zip(rgb_files, depth_files, gt):
            self._append_frame(rgb_path, depth_path, ts_key(rgb_path), row[1:4], row[4:8])

    def _load_full(self) -> None:
        """Full streams associated TUM-style by nearest timestamp."""
        rgb = read_file_list(os.path.join(self.seq_dir, "rgb.txt"))
        depth = read_file_list(os.path.join(self.seq_dir, "depth.txt"))
        gt = read_file_list(os.path.join(self.seq_dir, "groundtruth.txt"))
        gt_ts = sorted(gt)
        for t_rgb, t_dep in associate(list(rgb), list(depth), self.assoc_max_diff):
            t_gt = min(gt_ts, key=lambda g: abs(g - t_rgb))
            if abs(t_gt - t_rgb) > self.assoc_max_diff:
                continue
            vals = [float(v) for v in gt[t_gt]]
            self._append_frame(
                os.path.join(self.seq_dir, rgb[t_rgb][0]),
                os.path.join(self.seq_dir, depth[t_dep][0]),
                t_rgb,
                vals[0:3],
                vals[3:7],
            )

    def load_intrinsics(self) -> None:
        fx, fy, cx, cy = self._intrinsics_override or self._BONN_INTRINSICS
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
        return float(self._frames[int(frame_id)][2])

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        with Image.open(self._frames[int(frame_id)][0]) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Bonn provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Bonn provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """16-bit depth PNG -> ``(H, W)`` float32 metres (value / depth_scale)."""
        with Image.open(self._frames[int(frame_id)][1]) as im:
            arr = np.asarray(im).astype(np.float32)
        depth = arr / self.depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("Bonn provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (color camera if pose_frame=camera)."""
        _, _, _, t_c2w, quat_xyzw = self._frames[int(frame_id)]
        return NumpySE3Pose.from_quat(quat_xyzw, t_c2w, scalar_last=True)

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
        raise NotImplementedError("Bonn provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("Bonn provides no ground-truth point cloud")

    # -- manifest-backed path lookup ----------------------------------------- #
    def _frame_image_path(self, sensor_id, frame_id) -> str:
        return self._frames[int(frame_id)][0]

    # parse() is inherited from BaseSequence (concrete template over the getters).

    def __repr__(self) -> str:
        return f"BonnSequence(seq_id={self.seq_id!r}, subset={self.subset!r}, frames={len(self._frames)})"
