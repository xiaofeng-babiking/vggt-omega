"""KITTI (depth benchmark val drives) vendor for the VGGT-Omega dataset API.

This copy is the KITTI depth prediction/completion benchmark layout. Only the
``val/`` drives are usable as RGB-D video; they are full KITTI-raw recordings::

    val/<date>_drive_XXXX_sync/          13 drives across 5 recording dates
    |-- image_0{0..3}/data/NNNNNNNNNN.png   RGB (02/03 color, 00/01 grayscale)
    |-- image_0X/timestamps.txt             per-camera capture clock, one line/frame
    |-- oxts/{data/NNNNNNNNNN.txt, timestamps.txt, dataformat.txt}  GPS/IMU
    |-- velodyne_points/...                  raw LiDAR (unused here)
    `-- proj_depth/groundtruth/image_0{2,3}/NNNNNNNNNN.png  GT depth (uint16)

``train/`` drives ship ONLY ``proj_depth/groundtruth`` (no RGB / oxts /
timestamps on disk), so they cannot be loaded as image+depth sequences and are
excluded. ``depth_selection/val_selection_cropped`` (the official 1000-frame
single-image protocol: per-image K on disk, no poses, no temporal structure)
is intentionally NOT supported by this vendor -- it would need single-frame
sequences without EXTRINSICS/TIMESTAMP and a different AVAILABLE set; use a
dedicated loader if that protocol is ever needed.

One sequence = one drive x one color camera (cam 02 / cam 03, the rectified
color stereo pair); CAMERA_ID is advertised. The frame list is restricted to
frames that HAVE GT depth: the benchmark provides depth only for frame indices
5..N-6 of every drive (the first/last 5 frames lack accumulated LiDAR).

Conventions used here (validated empirically against this copy, not assumed):

* Depth PNG is uint16; ``meters = value / 256.0``; 0 = no LiDAR return
  (invalid). Depth is SPARSE projected LiDAR (~16-23% of pixels valid);
  POINT_MASK (= depth > 0) does the masking work. Sky has NO special encoding
  (it is 0 like any other no-return pixel), so SKY_MASK is neither advertised
  nor emitted -- outdoor driving data does contain sky, and an all-False mask
  would be wrong GT (same rule as the Waymo vendor).
* Poses come from ``oxts`` GPS/IMU via the standard pykitti-style chain:
  lat/lon -> local mercator (scale fixed from frame 0's latitude per drive),
  ``T_w_imu = [Rz(yaw)Ry(pitch)Rx(roll) | t]``, then
  ``c2w_camX = T_w_imu @ inv(T_camX_imu)`` with
  ``T_camX_imu = T_X @ R_rect00 @ T_cam0_velo @ T_velo_imu`` (T_X is the
  rectified-baseline shift ``P_rect_0X[0,3]/P_rect_0X[0,0]``). Camera axes are
  OpenCV (x right, y down, z forward); world->camera is the rigid inverse.
  VERIFIED by cross-frame depth reprojection on one drive of EVERY recording
  date, both cameras: median relative depth error 0.002-0.006; the flipped
  (c2w-as-w2c) convention fails at 0.45-3.0.
* The calibration files (``calib_cam_to_cam.txt`` etc.) are NOT on disk in
  this copy, so the per-date calib is hardcoded below from the official KITTI
  raw devkit calibration zips (one calib per recording date; all 5 dates in
  ``val/`` are covered and reprojection-verified as above). Drives from an
  unknown date are skipped with a warning.
* Intrinsics: ``K = P_rect_0X[:3,:3]`` of the date's calib, valid for the
  native full-resolution rectified image (resolution differs per date, e.g.
  1242x375 for 2011_09_26). Identical for cams 02/03 by rectification design.
* The mercator world frame has ~1e6 m coordinates, which would quantize to
  ~6 cm under float32; poses are therefore **recentered per drive**
  (subtracting frame 0's IMU position) before the float32 cast -- both
  camera-sequences of a drive share one consistent recentered world frame.
* TIMESTAMP comes from the real on-disk per-camera capture clock
  (``image_0X/timestamps.txt``, ~10 Hz, nanosecond precision), parsed to
  absolute epoch seconds (float64).

WORLD_POINTS / CAM_POINTS are only the depth re-projected through the
oxts-derived poses (not an independent point-cloud GT), so they are NOT
advertised as evaluable GT modalities -- process_one_image still computes
them, they just must not be scored as a point cloud.
"""
from __future__ import annotations

import calendar
import logging
import os
import random
import re
from datetime import datetime
from fnmatch import fnmatch

import cv2
import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class KittiDataset(BaseDataset):
    """KITTI val drives as a VGGT-Omega BaseDataset (driving video, sparse metric LiDAR depth)."""

    # Earth radius used by the KITTI devkit's mercator projection (meters).
    _EARTH_RADIUS = 6378137.0

    # GT depth / RGB frame filenames are zero-padded 10-digit indices.
    _FRAME_RE = re.compile(r"^(\d{10})\.png$")

    # Color cameras of the rectified stereo pair that have GT depth.
    _ALL_CAMERAS = (2, 3)

    # Official KITTI raw devkit calibration, hardcoded per recording date
    # because no calib files ship with this copy (see module docstring).
    # P_rect_0X: (3,4) rectified projection; R_rect_00: (3,3) rectifying
    # rotation; R/T_velo_cam0: velodyne->cam0 rigid; R/T_imu_velo: imu->velodyne
    # rigid. All values verified by cross-frame depth reprojection (see module
    # docstring).
    _DATE_CALIB = {
        "2011_09_26": {
            "P_rect_02": [7.215377e02, 0.0, 6.095593e02, 4.485728e01,
                          0.0, 7.215377e02, 1.728540e02, 2.163791e-01,
                          0.0, 0.0, 1.0, 2.745884e-03],
            "P_rect_03": [7.215377e02, 0.0, 6.095593e02, -3.395242e02,
                          0.0, 7.215377e02, 1.728540e02, 2.199936e00,
                          0.0, 0.0, 1.0, 2.729905e-03],
            "R_rect_00": [9.999239e-01, 9.837760e-03, -7.445048e-03,
                          -9.869795e-03, 9.999421e-01, -4.278459e-03,
                          7.402527e-03, 4.351614e-03, 9.999631e-01],
            "R_velo_cam0": [7.533745e-03, -9.999714e-01, -6.166020e-04,
                            1.480249e-02, 7.280733e-04, -9.998902e-01,
                            9.998621e-01, 7.523790e-03, 1.480755e-02],
            "T_velo_cam0": [-4.069766e-03, -7.631618e-02, -2.717806e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03,
                           -7.854027e-04, 9.998898e-01, -1.482298e-02,
                           2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
        "2011_09_28": {
            "P_rect_02": [7.070493e02, 0.0, 6.040814e02, 4.575831e01,
                          0.0, 7.070493e02, 1.805066e02, -3.454157e-01,
                          0.0, 0.0, 1.0, 4.981016e-03],
            "P_rect_03": [7.070493e02, 0.0, 6.040814e02, -3.341081e02,
                          0.0, 7.070493e02, 1.805066e02, 2.330660e00,
                          0.0, 0.0, 1.0, 3.201153e-03],
            "R_rect_00": [9.999128e-01, 1.009263e-02, -8.511932e-03,
                          -1.012729e-02, 9.999406e-01, -4.037671e-03,
                          8.470675e-03, 4.123522e-03, 9.999556e-01],
            "R_velo_cam0": [6.927964e-03, -9.999722e-01, -2.757829e-03,
                            -1.162982e-03, 2.749836e-03, -9.999955e-01,
                            9.999753e-01, 6.931141e-03, -1.143899e-03],
            "T_velo_cam0": [-2.457729e-02, -6.127237e-02, -3.321029e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03,
                           -7.854027e-04, 9.998898e-01, -1.482298e-02,
                           2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
        "2011_09_29": {
            "P_rect_02": [7.183351e02, 0.0, 6.003891e02, 4.450382e01,
                          0.0, 7.183351e02, 1.815122e02, -5.951107e-01,
                          0.0, 0.0, 1.0, 2.616315e-03],
            "P_rect_03": [7.183351e02, 0.0, 6.003891e02, -3.363147e02,
                          0.0, 7.183351e02, 1.815122e02, 3.159867e00,
                          0.0, 0.0, 1.0, 5.323834e-03],
            "R_rect_00": [9.999478e-01, 9.791707e-03, -2.925305e-03,
                          -9.806939e-03, 9.999382e-01, -5.238719e-03,
                          2.873828e-03, 5.267134e-03, 9.999820e-01],
            "R_velo_cam0": [7.755449e-03, -9.999694e-01, -1.014303e-03,
                            2.294056e-03, 1.032122e-03, -9.999968e-01,
                            9.999673e-01, 7.753097e-03, 2.301990e-03],
            "T_velo_cam0": [-7.275538e-03, -6.324057e-02, -2.670414e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03,
                           -7.854027e-04, 9.998898e-01, -1.482298e-02,
                           2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
        "2011_09_30": {
            "P_rect_02": [7.070912e02, 0.0, 6.018873e02, 4.688783e01,
                          0.0, 7.070912e02, 1.831104e02, 1.178601e-01,
                          0.0, 0.0, 1.0, 6.203223e-03],
            "P_rect_03": [7.070912e02, 0.0, 6.018873e02, -3.334597e02,
                          0.0, 7.070912e02, 1.831104e02, 1.930130e00,
                          0.0, 0.0, 1.0, 3.318498e-03],
            "R_rect_00": [9.999280e-01, 8.085985e-03, -8.866797e-03,
                          -8.123205e-03, 9.999583e-01, -4.169750e-03,
                          8.832711e-03, 4.241477e-03, 9.999520e-01],
            "R_velo_cam0": [7.027555e-03, -9.999753e-01, 2.599616e-05,
                            -2.254837e-03, -4.184312e-05, -9.999975e-01,
                            9.999728e-01, 7.027479e-03, -2.255075e-03],
            "T_velo_cam0": [-7.137748e-03, -7.482656e-02, -3.336324e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03,
                           -7.854027e-04, 9.998898e-01, -1.482298e-02,
                           2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
        "2011_10_03": {
            "P_rect_02": [7.188560e02, 0.0, 6.071928e02, 4.538225e01,
                          0.0, 7.188560e02, 1.852157e02, -1.130887e-01,
                          0.0, 0.0, 1.0, 3.779761e-03],
            "P_rect_03": [7.188560e02, 0.0, 6.071928e02, -3.372877e02,
                          0.0, 7.188560e02, 1.852157e02, 2.369057e00,
                          0.0, 0.0, 1.0, 4.915215e-03],
            "R_rect_00": [9.999454e-01, 7.259129e-03, -7.519551e-03,
                          -7.292213e-03, 9.999638e-01, -4.381729e-03,
                          7.487471e-03, 4.436324e-03, 9.999621e-01],
            "R_velo_cam0": [7.967514e-03, -9.999679e-01, -8.462264e-04,
                            -2.771053e-03, 8.241710e-04, -9.999958e-01,
                            9.999644e-01, 7.969825e-03, -2.764397e-03],
            "T_velo_cam0": [-1.377769e-02, -5.542117e-02, -2.918589e-01],
            "R_imu_velo": [9.999976e-01, 7.553071e-04, -2.035826e-03,
                           -7.854027e-04, 9.998898e-01, -1.482298e-02,
                           2.024406e-03, 1.482454e-02, 9.998881e-01],
            "T_imu_velo": [-8.086759e-01, 3.195559e-01, -7.997231e-01],
        },
    }

    # KITTI provides RGB + sparse metric LiDAR depth + oxts-derived GT poses,
    # per-date devkit intrinsics, real per-camera capture timestamps and two
    # color cameras per drive (CAMERA_ID). No SKY_MASK (sparse LiDAR cannot
    # separate sky from missing returns; no "sky_masks" key is emitted either
    # -- an all-False mask for outdoor driving data would be wrong GT, same
    # rule as the Waymo vendor). WORLD_POINTS / CAM_POINTS are never
    # advertised (see module docstring).
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.POINT_MASK,
            Modality.TIMESTAMP,
            Modality.CAMERA_ID,
        }
    )

    # --- pure helpers (unit-testable without data) -------------------------

    @staticmethod
    def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """oxts roll/pitch/yaw (radians) -> (3,3) rotation ``Rz(yaw)Ry(pitch)Rx(roll)``
        (the KITTI devkit/pykitti convention; yaw=0 means east, CCW positive)."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return rz @ ry @ rx

    @staticmethod
    def mercator_scale(lat0_deg: float) -> float:
        """Per-drive mercator scale from the REFERENCE frame's latitude (frame 0
        of the drive; the scale must stay fixed across the whole drive)."""
        return float(np.cos(lat0_deg * np.pi / 180.0))

    @classmethod
    def oxts_to_imu_pose(cls, oxts_vals, scale: float) -> np.ndarray:
        """oxts record (>= 6 floats: lat lon alt roll pitch yaw ...) -> (4,4)
        float64 ``T_w_imu`` via the devkit mercator projection. Raises
        ValueError on a short or non-finite record."""
        o = np.asarray(oxts_vals, dtype=np.float64).ravel()
        if o.size < 6:
            raise ValueError(f"oxts record: expected >= 6 values, got {o.size}")
        lat, lon, alt, roll, pitch, yaw = o[:6]
        if not np.isfinite(o[:6]).all():
            raise ValueError("oxts record is non-finite")
        er = cls._EARTH_RADIUS
        t = np.array(
            [
                scale * lon * np.pi / 180.0 * er,
                scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0)),
                alt,
            ]
        )
        pose = np.eye(4)
        pose[:3, :3] = cls.rotation_from_rpy(roll, pitch, yaw)
        pose[:3, 3] = t
        return pose

    @classmethod
    def cam_from_imu(cls, date: str, cam: int) -> np.ndarray:
        """(4,4) float64 ``T_camX_imu`` (IMU -> rectified color camera ``cam``)
        from the hardcoded per-date devkit calib:
        ``T_X @ R_rect00 @ T_cam0_velo @ T_velo_imu`` with the rectified
        baseline shift ``T_X[0,3] = P_rect_0X[0,3] / P_rect_0X[0,0]``.
        Raises ValueError for an unknown date or camera."""
        calib = cls._DATE_CALIB.get(date)
        if calib is None:
            raise ValueError(
                f"no hardcoded KITTI devkit calib for date {date!r}; "
                f"known dates: {sorted(cls._DATE_CALIB)}"
            )
        if cam not in cls._ALL_CAMERAS:
            raise ValueError(f"cam must be one of {cls._ALL_CAMERAS}, got {cam}")
        p_rect = np.asarray(calib[f"P_rect_0{cam}"], dtype=np.float64).reshape(3, 4)
        r_rect = np.eye(4)
        r_rect[:3, :3] = np.asarray(calib["R_rect_00"], dtype=np.float64).reshape(3, 3)
        t_cam0_velo = np.eye(4)
        t_cam0_velo[:3, :3] = np.asarray(calib["R_velo_cam0"], dtype=np.float64).reshape(3, 3)
        t_cam0_velo[:3, 3] = np.asarray(calib["T_velo_cam0"], dtype=np.float64)
        t_velo_imu = np.eye(4)
        t_velo_imu[:3, :3] = np.asarray(calib["R_imu_velo"], dtype=np.float64).reshape(3, 3)
        t_velo_imu[:3, 3] = np.asarray(calib["T_imu_velo"], dtype=np.float64)
        t_x = np.eye(4)
        t_x[0, 3] = p_rect[0, 3] / p_rect[0, 0]
        return t_x @ r_rect @ t_cam0_velo @ t_velo_imu

    @staticmethod
    def imu_pose_to_w2c(t_w_imu, t_cam_imu, anchor=None) -> np.ndarray:
        """``T_w_imu`` (4,4) + ``T_camX_imu`` (4,4) -> world-to-camera (3,4)
        float32 OpenCV. ``anchor`` (3,) is subtracted from the IMU position
        first to recenter the ~1e6 m mercator coordinates before the float32
        cast. Raises ValueError on non-finite input."""
        t_w_imu = np.asarray(t_w_imu, dtype=np.float64)
        t_cam_imu = np.asarray(t_cam_imu, dtype=np.float64)
        if t_w_imu.shape != (4, 4) or not np.isfinite(t_w_imu).all():
            raise ValueError("T_w_imu must be a finite (4,4) matrix")
        if anchor is not None:
            t_w_imu = t_w_imu.copy()
            t_w_imu[:3, 3] -= np.asarray(anchor, dtype=np.float64).reshape(3)
        c2w = t_w_imu @ np.linalg.inv(t_cam_imu)
        return np.linalg.inv(c2w)[:3, :4].astype(np.float32)

    @classmethod
    def kitti_intrinsics(cls, date: str, cam: int = 2, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K at the NATIVE full resolution of the date's
        rectified images: ``P_rect_0{cam}[:3,:3]`` from the hardcoded devkit
        calib. ``override``=[fx, fy, cx, cy] wins. Raises ValueError for an
        unknown date/camera."""
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        calib = cls._DATE_CALIB.get(date)
        if calib is None:
            raise ValueError(
                f"no hardcoded KITTI devkit calib for date {date!r}; "
                f"known dates: {sorted(cls._DATE_CALIB)}"
            )
        if cam not in cls._ALL_CAMERAS:
            raise ValueError(f"cam must be one of {cls._ALL_CAMERAS}, got {cam}")
        p_rect = np.asarray(calib[f"P_rect_0{cam}"], dtype=np.float64).reshape(3, 4)
        return p_rect[:3, :3].astype(np.float32)

    @staticmethod
    def read_kitti_depth(path: str) -> np.ndarray:
        """Read a KITTI benchmark GT depth PNG -> float32 (H,W) meters.

        Stored as uint16 counts; ``meters = value / 256.0``; 0 = no LiDAR
        return (invalid, includes all sky). Raises FileNotFoundError when the
        PNG cannot be read."""
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"KITTI: could not read depth PNG {path}")
        depth = arr.astype(np.float32) / 256.0
        depth[~np.isfinite(depth)] = 0.0
        return depth

    @staticmethod
    def parse_kitti_timestamp(line: str) -> float:
        """KITTI timestamps.txt line ('2011-09-26 13:02:44.335092332', ns
        precision) -> absolute epoch seconds (float). Raises ValueError on a
        blank/malformed line."""
        line = line.strip()
        if not line:
            raise ValueError("empty KITTI timestamp line")
        base, _, frac = line.partition(".")
        dt = datetime.strptime(base, "%Y-%m-%d %H:%M:%S")
        secs = float(calendar.timegm(dt.timetuple()))
        if frac:
            secs += float("0." + frac)
        return secs

    @classmethod
    def parse_gt_depth_listing(cls, filenames) -> list:
        """GT-depth directory listing -> sorted list of int frame indices.
        Only ``NNNNNNNNNN.png`` entries count; stray files are ignored."""
        out = []
        for fn in filenames:
            m = cls._FRAME_RE.match(fn)
            if m:
                out.append(int(m.group(1)))
        out.sort()
        return out

    @staticmethod
    def drive_date(drive: str) -> str:
        """Drive directory name ('2011_09_26_drive_0002_sync') -> recording
        date ('2011_09_26', the per-date calib key)."""
        return "_".join(drive.split("_")[:3])

    # --- construction -------------------------------------------------------

    def __init__(
        self,
        common_conf,
        split: str = "val",
        KITTI_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        cameras=(2, 3),
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if KITTI_DIR is None:
            raise ValueError("KITTI_DIR must be specified")
        cameras = tuple(int(c) for c in cameras)
        if not cameras or any(c not in self._ALL_CAMERAS for c in cameras):
            raise ValueError(
                f"cameras must be a non-empty subset of {self._ALL_CAMERAS}, got {cameras}"
            )

        self.KITTI_DIR = KITTI_DIR
        # Only val/ ships RGB-D video (train/ is GT-depth-only, see module
        # docstring); `split` only selects the virtual epoch length.
        val_root = os.path.join(KITTI_DIR, "val")
        if not os.path.isdir(val_root):
            val_root = KITTI_DIR  # KITTI_DIR may point directly at the drive dirs
        self.val_root = val_root
        self.expand_ratio = expand_ratio
        self.cameras = cameras
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        patterns = list(sequences) if sequences else None
        try:
            drive_dirs = sorted(
                (e for e in os.scandir(val_root) if e.is_dir()),
                key=lambda e: e.name,
            )
        except FileNotFoundError:
            raise ValueError(f"KITTI drive root {val_root} does not exist")

        self._drive_dirs = {}  # drive name -> absolute drive dir
        self.data_store = {}   # seq_name -> [(rgb_path, depth_path, oxts_path, frame_idx)]
        for entry in drive_dirs:
            drive = entry.name
            date = self.drive_date(drive)
            if date not in self._DATE_CALIB:
                logging.warning(
                    "KITTI drive %s: no hardcoded devkit calib for date %s; skipping",
                    drive, date,
                )
                continue
            for cam in cameras:
                seq_name = f"{drive}/cam{cam:02d}"
                if patterns is not None and not any(
                    fnmatch(seq_name, p) or fnmatch(drive, p) for p in patterns
                ):
                    continue
                gt_dir = os.path.join(entry.path, "proj_depth", "groundtruth", f"image_0{cam}")
                if not os.path.isdir(gt_dir):
                    logging.warning("KITTI %s: no GT depth dir %s; skipping", seq_name, gt_dir)
                    continue
                frame_ids = self.parse_gt_depth_listing(os.listdir(gt_dir))
                if len(frame_ids) < min_num_images:
                    logging.warning(
                        "KITTI %s: only %d GT-depth frames (< %d); skipping",
                        seq_name, len(frame_ids), min_num_images,
                    )
                    continue
                rgb_dir = os.path.join(entry.path, f"image_0{cam}", "data")
                oxts_dir = os.path.join(entry.path, "oxts", "data")
                self.data_store[seq_name] = [
                    (
                        os.path.join(rgb_dir, f"{n:010d}.png"),
                        os.path.join(gt_dir, f"{n:010d}.png"),
                        os.path.join(oxts_dir, f"{n:010d}.txt"),
                        n,
                    )
                    for n in frame_ids
                ]
                self._drive_dirs[drive] = entry.path

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable KITTI sequences under {val_root} "
                f"(sequences={patterns}, cameras={cameras})"
            )

        self._scale_anchor_cache = {}  # drive -> (mercator scale, (3,) anchor)
        self._timestamps_cache = {}    # (drive, cam) -> float64 array, line i = frame i
        self._native_size_cache = {}

    # --- lazy per-drive metadata --------------------------------------------

    @staticmethod
    def _read_oxts(path: str) -> np.ndarray:
        """Read one oxts record -> float64 array (plain split: ~100x faster
        than np.loadtxt per file)."""
        with open(path) as f:
            return np.asarray(f.read().split(), dtype=np.float64)

    def _drive_scale_anchor(self, drive: str):
        """Per-drive (mercator scale, recentering anchor): both fixed from the
        drive's frame-0 oxts record (falling back to the drive's first
        GT-depth frame if frame 0 is missing), lazily read and cached. The
        anchor is the reference frame's IMU mercator position; both
        camera-sequences of the drive share it (one consistent world frame)."""
        if drive not in self._scale_anchor_cache:
            ref_path = os.path.join(self._drive_dirs[drive], "oxts", "data", "0000000000.txt")
            if not os.path.isfile(ref_path):
                ref_path = next(
                    frames[0][2]
                    for name, frames in self.data_store.items()
                    if name.startswith(drive + "/")
                )
            oxts = self._read_oxts(ref_path)
            scale = self.mercator_scale(oxts[0])
            anchor = self.oxts_to_imu_pose(oxts, scale)[:3, 3].copy()
            self._scale_anchor_cache[drive] = (scale, anchor)
        return self._scale_anchor_cache[drive]

    def _frame_timestamps(self, drive: str, cam: int) -> np.ndarray:
        """Per-camera capture clock for one drive: ``image_0X/timestamps.txt``
        parsed to absolute epoch seconds (float64), line i = frame index i.
        Lazily read and cached."""
        key = (drive, cam)
        if key not in self._timestamps_cache:
            path = os.path.join(self._drive_dirs[drive], f"image_0{cam}", "timestamps.txt")
            with open(path) as f:
                ts = np.array(
                    [self.parse_kitti_timestamp(line) for line in f if line.strip()],
                    dtype=np.float64,
                )
            self._timestamps_cache[key] = ts
        return self._timestamps_cache[key]

    # --- BaseDataset contract -------------------------------------------------

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of GT-depth frames in the sequence at ``local_idx`` of this
        vendor's ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached
        (resolution differs per recording date, e.g. 375x1242 for 2011_09_26)."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self.data_store[name][0][0]
            with Image.open(rgb_path) as im:
                w, h = im.size  # PIL reports (W, H) without decoding pixels
            self._native_size_cache[name] = (h, w)
        return self._native_size_cache[name]

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids=None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        if seq_name is None:
            # len_train is a virtual epoch length usually >> #sequences; the sampler
            # emits indices in [0, len_train). That only works with inside_random=True
            # (which remapped seq_index above). Give a clear error, not a raw IndexError.
            if seq_index is None or not 0 <= seq_index < self.sequence_list_len:
                raise ValueError(
                    f"seq_index={seq_index} out of range [0, {self.sequence_list_len}); "
                    "when len_train > #sequences, set inside_random=True so the sampler "
                    "index is remapped into range."
                )
            seq_name = self.sequence_list[seq_index]
        frames = self.data_store[seq_name]
        drive, cam_token = seq_name.rsplit("/", 1)
        cam = int(cam_token.replace("cam", ""))
        date = self.drive_date(drive)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)
        K = self.kitti_intrinsics(date, cam, self.intrinsics_override)
        t_cam_imu = self.cam_from_imu(date, cam)
        scale, anchor = self._drive_scale_anchor(drive)
        frame_ts = self._frame_timestamps(drive, cam)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        camera_ids, timestamps, original_sizes = [], [], []

        for i in ids:
            rgb_path, depth_path, oxts_path, frame_idx = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Enumerated from the GT-depth listing, so the file should
                # exist; fail loudly (a silent skip would yield fewer than
                # img_per_seq frames and break fixed-V batch stacking).
                raise FileNotFoundError(f"KITTI: could not read image {rgb_path}")
            depth_map = self.read_kitti_depth(depth_path)
            if depth_map.shape != image.shape[:2]:
                raise ValueError(
                    f"KITTI: depth {depth_map.shape} does not match image "
                    f"{image.shape[:2]} for {depth_path}"
                )
            t_w_imu = self.oxts_to_imu_pose(self._read_oxts(oxts_path), scale)
            pose_w2c = self.imu_pose_to_w2c(t_w_imu, t_cam_imu, anchor=anchor)
            if frame_idx >= len(frame_ts):
                raise ValueError(
                    f"KITTI {seq_name}: frame {frame_idx} has no line in "
                    f"image_0{cam}/timestamps.txt ({len(frame_ts)} lines)"
                )
            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri,
                intri,
                world_p,
                cam_p,
                pmask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                pose_w2c.copy(),
                K.copy(),
                original_size,
                target_image_shape,
                filepath=rgb_path,
            )

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri)
            intrinsics.append(intri)
            cam_points.append(cam_p)
            world_points.append(world_p)
            point_masks.append(pmask)
            camera_ids.append(cam)
            timestamps.append(frame_ts[frame_idx])
            original_sizes.append(original_size)

        return {
            "seq_name": "kitti_" + seq_name,
            "ids": np.array(ids),
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            # NO "sky_masks": SKY_MASK is not advertised and an all-False mask
            # would be wrong GT for outdoor driving data (sky lives in depth==0).
            "camera_ids": np.array(camera_ids, dtype=np.int32),
            "timestamps": np.array(timestamps, dtype=np.float64),
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
