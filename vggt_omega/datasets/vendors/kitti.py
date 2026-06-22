"""KITTI (depth benchmark val drives) vendor, implemented against :class:`BaseSequence`.

KITTI-raw recordings with benchmark GT depth::

    {data_root}/val/<date>_drive_XXXX_sync/
        image_0{2,3}/data/NNNNNNNNNN.png             RGB (02/03 colour)
        image_0X/timestamps.txt                      per-camera clock
        oxts/data/NNNNNNNNNN.txt, oxts/timestamps.txt  GPS/IMU
        proj_depth/groundtruth/image_0{2,3}/*.png    GT depth (uint16)

Conventions (anchored to the original ``KittiDataset`` loader):

* One sequence = one drive x one colour camera (``camera`` in {2, 3}). The frame
  list is restricted to frames that HAVE GT depth (benchmark provides depth for
  indices 5..N-6).
* Depth: uint16 PNG, ``metres = value / 256``; 0 = no LiDAR return (invalid,
  sparse ~16-23%). No sky encoding.
* Pose: derived from ``oxts`` via the pykitti chain (mercator world frame, scale
  fixed from frame 0's latitude), recentred per drive; :meth:`get_pose` returns
  the **camera-to-world** OpenCV pose.
* Intrinsics: hardcoded per-date devkit calib ``P_rect_0{cam}[:3,:3]``; override
  via ``intrinsics=(fx, fy, cx, cy)``.

A :class:`BaseSequence` is one ``<drive>`` + ``camera``; ``seq_id`` is the drive
directory name. TIMESTAMP comes from ``image_0X/timestamps.txt`` (epoch seconds).
"""
from __future__ import annotations

import calendar
import os
import re
from datetime import datetime
from typing import List, Optional, Set, Tuple, Union

import cv2
import numpy as np

from vggt_omega.datasets.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose


class KittiSequence(BaseSequence):
    """One KITTI drive x colour camera as a :class:`BaseSequence`."""

    SENSOR: int = 0
    _EARTH_RADIUS = 6378137.0
    _FRAME_RE = re.compile(r"^(\d{10})\.png$")
    _ALL_CAMERAS = (2, 3)

    # Official KITTI raw devkit calibration, hardcoded per recording date.
    _DATE_CALIB = {
        "2011_09_26": {
            "P_rect_02": [7.215377e02, 0.0, 6.095593e02, 4.485728e01, 0.0, 7.215377e02, 1.728540e02, 2.163791e-01, 0.0, 0.0, 1.0, 2.745884e-03],
            "P_rect_03": [7.215377e02, 0.0, 6.095593e02, -3.395242e02, 0.0, 7.215377e02, 1.728540e02, 2.199936e00, 0.0, 0.0, 1.0, 2.729905e-03],
            "R_rect_00": [9.999239e-01, 9.837760e-03, -7.445048e-03, -9.869795e-03, 9.999421e-01, -4.278459e-03, 7.402527e-03, 4.351614e-03, 9.999631e-01],
            "R_velo_cam0": [7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04, -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02],
            "T_velo_cam0": [-4.069766e-03, -7.631618e-02, -2.717806e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03, -7.854027e-04, 9.998898e-01, -1.482298e-02, 2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
        "2011_09_28": {
            "P_rect_02": [7.070493e02, 0.0, 6.040814e02, 4.575831e01, 0.0, 7.070493e02, 1.805066e02, -3.454157e-01, 0.0, 0.0, 1.0, 4.981016e-03],
            "P_rect_03": [7.070493e02, 0.0, 6.040814e02, -3.341081e02, 0.0, 7.070493e02, 1.805066e02, 2.330660e00, 0.0, 0.0, 1.0, 3.201153e-03],
            "R_rect_00": [9.999128e-01, 1.009263e-02, -8.511932e-03, -1.012729e-02, 9.999406e-01, -4.037671e-03, 8.470675e-03, 4.123522e-03, 9.999556e-01],
            "R_velo_cam0": [6.927964e-03, -9.999722e-01, -2.757829e-03, -1.162982e-03, 2.749836e-03, -9.999955e-01, 9.999753e-01, 6.931141e-03, -1.143899e-03],
            "T_velo_cam0": [-2.457729e-02, -6.127237e-02, -3.321029e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03, -7.854027e-04, 9.998898e-01, -1.482298e-02, 2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
        "2011_09_29": {
            "P_rect_02": [7.183351e02, 0.0, 6.003891e02, 4.450382e01, 0.0, 7.183351e02, 1.815122e02, -5.951107e-01, 0.0, 0.0, 1.0, 2.616315e-03],
            "P_rect_03": [7.183351e02, 0.0, 6.003891e02, -3.363147e02, 0.0, 7.183351e02, 1.815122e02, 3.159867e00, 0.0, 0.0, 1.0, 5.323834e-03],
            "R_rect_00": [9.999478e-01, 9.791707e-03, -2.925305e-03, -9.806939e-03, 9.999382e-01, -5.238719e-03, 2.873828e-03, 5.267134e-03, 9.999820e-01],
            "R_velo_cam0": [7.755449e-03, -9.999694e-01, -1.014303e-03, 2.294056e-03, 1.032122e-03, -9.999968e-01, 9.999673e-01, 7.753097e-03, 2.301990e-03],
            "T_velo_cam0": [-7.275538e-03, -6.324057e-02, -2.670414e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03, -7.854027e-04, 9.998898e-01, -1.482298e-02, 2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
        "2011_09_30": {
            "P_rect_02": [7.070912e02, 0.0, 6.018873e02, 4.688783e01, 0.0, 7.070912e02, 1.831104e02, 1.178601e-01, 0.0, 0.0, 1.0, 6.203223e-03],
            "P_rect_03": [7.070912e02, 0.0, 6.018873e02, -3.334597e02, 0.0, 7.070912e02, 1.831104e02, 1.930130e00, 0.0, 0.0, 1.0, 3.318498e-03],
            "R_rect_00": [9.999280e-01, 8.085985e-03, -8.866797e-03, -8.123205e-03, 9.999583e-01, -4.169750e-03, 8.832711e-03, 4.241477e-03, 9.999520e-01],
            "R_velo_cam0": [7.027555e-03, -9.999753e-01, 2.599616e-05, -2.254837e-03, -4.184312e-05, -9.999975e-01, 9.999728e-01, 7.027479e-03, -2.255075e-03],
            "T_velo_cam0": [-7.137748e-03, -7.482656e-02, -3.336324e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03, -7.854027e-04, 9.998898e-01, -1.482298e-02, 2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
        "2011_10_03": {
            "P_rect_02": [7.188560e02, 0.0, 6.071928e02, 4.538225e01, 0.0, 7.188560e02, 1.852157e02, -1.130887e-01, 0.0, 0.0, 1.0, 3.779761e-03],
            "P_rect_03": [7.188560e02, 0.0, 6.071928e02, -3.372877e02, 0.0, 7.188560e02, 1.852157e02, 2.369057e00, 0.0, 0.0, 1.0, 4.915215e-03],
            "R_rect_00": [9.999454e-01, 7.259129e-03, -7.519551e-03, -7.292213e-03, 9.999638e-01, -4.381729e-03, 7.487471e-03, 4.436324e-03, 9.999621e-01],
            "R_velo_cam0": [7.967514e-03, -9.999679e-01, -8.462264e-04, -2.771053e-03, 8.241710e-04, -9.999958e-01, 9.999644e-01, 7.969825e-03, -2.764397e-03],
            "T_velo_cam0": [-1.377769e-02, -5.542117e-02, -2.918589e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03, -7.854027e-04, 9.998898e-01, -1.482298e-02, 2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
    }

    _MODALITIES = frozenset(
        {Modality.RGB, Modality.DEPTH, Modality.POSE, Modality.TIMESTAMP, Modality.INTRINSIC, Modality.EXTRINSIC}
    )

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        *,
        camera: int = 2,
        intrinsics: Optional[Tuple[float, float, float, float]] = None,
    ):
        if camera not in self._ALL_CAMERAS:
            raise ValueError(f"camera must be one of {self._ALL_CAMERAS}, got {camera}")
        self.data_root = data_root
        self.seq_id = seq_id
        self.camera = int(camera)
        val_root = os.path.join(data_root, "val")
        if not os.path.isdir(os.path.join(val_root, seq_id)):
            val_root = data_root
        self.seq_dir = os.path.join(val_root, seq_id)
        self.date = "_".join(seq_id.split("_")[:3])
        self._intrinsics_override = intrinsics

        self._frame_ids: List[int] = []          # GT-depth frame indices
        self._oxts: dict = {}                     # idx -> oxts vals
        self._cam_ts: dict = {}                   # idx -> epoch seconds
        self._scale: float = 1.0
        self._anchor: Optional[np.ndarray] = None
        self._t_cam_imu: Optional[np.ndarray] = None
        self._intrinsic: Optional[np.ndarray] = None

        self.load_manifest()
        self.load_extrinsics()
        self.load_intrinsics()

    # -- static calib / pose helpers (verbatim from the original loader) ---- #
    @staticmethod
    def _rpy(roll, pitch, yaw) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]])
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return rz @ ry @ rx

    def _oxts_to_imu_pose(self, vals) -> np.ndarray:
        o = np.asarray(vals, dtype=np.float64).ravel()
        lat, lon, alt, roll, pitch, yaw = o[:6]
        er = self._EARTH_RADIUS
        t = np.array([
            self._scale * lon * np.pi / 180.0 * er,
            self._scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0)),
            alt,
        ])
        pose = np.eye(4)
        pose[:3, :3] = self._rpy(roll, pitch, yaw)
        pose[:3, 3] = t
        return pose

    def _cam_from_imu(self) -> np.ndarray:
        calib = self._DATE_CALIB.get(self.date)
        if calib is None:
            raise ValueError(f"no KITTI calib for date {self.date!r}; known {sorted(self._DATE_CALIB)}")
        cam = self.camera
        p_rect = np.asarray(calib[f"P_rect_0{cam}"], dtype=np.float64).reshape(3, 4)
        r_rect = np.eye(4); r_rect[:3, :3] = np.asarray(calib["R_rect_00"], dtype=np.float64).reshape(3, 3)
        t_cam0_velo = np.eye(4)
        t_cam0_velo[:3, :3] = np.asarray(calib["R_velo_cam0"], dtype=np.float64).reshape(3, 3)
        t_cam0_velo[:3, 3] = np.asarray(calib["T_velo_cam0"], dtype=np.float64)
        t_velo_imu = np.eye(4)
        t_velo_imu[:3, :3] = np.asarray(calib["R_imu_velo"], dtype=np.float64).reshape(3, 3)
        t_velo_imu[:3, 3] = np.asarray(calib["T_imu_velo"], dtype=np.float64)
        t_x = np.eye(4); t_x[0, 3] = p_rect[0, 3] / p_rect[0, 0]
        return t_x @ r_rect @ t_cam0_velo @ t_velo_imu

    # -- lifecycle ----------------------------------------------------------- #
    def load_manifest(self) -> None:
        gt_dir = os.path.join(self.seq_dir, "proj_depth", "groundtruth", f"image_0{self.camera}")
        ids = []
        for fn in os.listdir(gt_dir):
            m = self._FRAME_RE.match(fn)
            if m:
                ids.append(int(m.group(1)))
        ids.sort()
        if not ids:
            raise ValueError(f"KITTI {self.seq_id} cam {self.camera}: no GT depth frames")
        self._frame_ids = ids

        # oxts records + per-camera timestamps for the listed frames.
        oxts_dir = os.path.join(self.seq_dir, "oxts", "data")
        for i in ids:
            with open(os.path.join(oxts_dir, f"{i:010d}.txt")) as f:
                self._oxts[i] = [float(v) for v in f.read().split()]
        ts_path = os.path.join(self.seq_dir, f"image_0{self.camera}", "timestamps.txt")
        with open(ts_path) as f:
            ts_lines = f.read().splitlines()
        for i in ids:
            self._cam_ts[i] = self._parse_ts(ts_lines[i])

    def load_extrinsics(self) -> None:
        # mercator scale fixed from the first listed frame's latitude; anchor =
        # that frame's IMU position to recentre the ~1e6 m world coordinates.
        lat0 = self._oxts[self._frame_ids[0]][0]
        self._scale = float(np.cos(lat0 * np.pi / 180.0))
        self._anchor = self._oxts_to_imu_pose(self._oxts[self._frame_ids[0]])[:3, 3].copy()
        self._t_cam_imu = self._cam_from_imu()

    def load_intrinsics(self) -> None:
        if self._intrinsics_override is not None:
            fx, fy, cx, cy = self._intrinsics_override
            self._intrinsic = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        else:
            p_rect = np.asarray(self._DATE_CALIB[self.date][f"P_rect_0{self.camera}"], dtype=np.float64).reshape(3, 4)
            self._intrinsic = p_rect[:3, :3].astype(np.float32)

    @staticmethod
    def _parse_ts(line: str) -> float:
        line = line.strip()
        base, _, frac = line.partition(".")
        secs = float(calendar.timegm(datetime.strptime(base, "%Y-%m-%d %H:%M:%S").timetuple()))
        return secs + (float("0." + frac) if frac else 0.0)

    # -- discovery ----------------------------------------------------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        return [self.SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return set(self._MODALITIES)

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._frame_ids)

    def get_timestamp(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> float:
        return float(self._cam_ts[self._frame_ids[int(frame_id)]])

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        i = self._frame_ids[int(frame_id)]
        path = os.path.join(self.seq_dir, f"image_0{self.camera}", "data", f"{i:010d}.png")
        from PIL import Image
        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_semantic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("KITTI provides no semantic masks")

    def get_dynamic_mask(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("KITTI provides no dynamic masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> np.ndarray:
        """Benchmark GT depth PNG -> float32 metres (value / 256); 0 = invalid."""
        i = self._frame_ids[int(frame_id)]
        path = os.path.join(self.seq_dir, "proj_depth", "groundtruth", f"image_0{self.camera}", f"{i:010d}.png")
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"KITTI: could not read depth {path}")
        depth = arr.astype(np.float32) / 256.0
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(self, sensor_id, frame_id) -> np.ndarray:
        raise NotImplementedError("KITTI provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]) -> BaseSE3Pose:
        """Per-frame **camera-to-world** SE(3) pose (recentred mercator world)."""
        i = self._frame_ids[int(frame_id)]
        t_w_imu = self._oxts_to_imu_pose(self._oxts[i])
        t_w_imu[:3, 3] -= self._anchor
        c2w = t_w_imu @ np.linalg.inv(self._t_cam_imu)
        return NumpySE3Pose.from_rot_mat(c2w[:3, :3], c2w[:3, 3])

    def get_extrinsic(self, src_sensor_id, dst_sensor_id) -> BaseSE3Pose:
        return NumpySE3Pose.identity(backend="numpy")

    def get_intrinsic(self, sensor_id: Union[int, str]) -> np.ndarray:
        assert self._intrinsic is not None
        return self._intrinsic.copy()

    def get_tracks(self, sensor_id) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("KITTI provides no 2D/3D tracks")

    def get_pointcloud(self) -> np.ndarray:
        raise NotImplementedError("KITTI provides no ground-truth point cloud")

    def _frame_image_path(self, sensor_id, frame_id) -> str:
        i = self._frame_ids[int(frame_id)]
        return os.path.join(self.seq_dir, f"image_0{self.camera}", "data", f"{i:010d}.png")

    def __repr__(self) -> str:
        return f"KittiSequence(seq_id={self.seq_id!r}, cam={self.camera}, frames={len(self._frame_ids)})"
