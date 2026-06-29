"""TUM RGB-D vendor, implemented against :class:`BaseSequence`.

A TUM RGB-D capture is a single moving RGB-D camera. The disk layout per
sequence directory is::

    <seq>/rgb.txt           # 'timestamp filename' index for the color frames
    <seq>/depth.txt         # 'timestamp filename' index for the depth frames
    <seq>/groundtruth.txt   # 'timestamp tx ty tz qx qy qz qw' camera-to-world
    <seq>/rgb/*.png         # 8-bit color
    <seq>/depth/*.png       # 16-bit depth, metres = pixel / 5000

RGB, depth and groundtruth run on three different clocks. RGB and depth are paired
by frame index (assumed pre-synchronized; the per-pair timestamp gap is asserted
to stay within ``rgbd_sync_diff``), and the groundtruth trajectory is interpolated
along the SE(3) geodesic onto the color timestamps
(``utils.se3_trajectory.NumpySE3Trajectory``) — which must therefore temporally
span them. Intrinsics are per-camera constants (freiburg1 / 2 / 3).

This class is a :class:`BaseSequence`: it models **one** sequence with **one**
sensor (``sensor_id == "RGBD"``, the RGB-D camera). Construction reads only the
text indices (no pixels); the ``get_rgb`` / ``get_depth`` getters decode lazily.
Per-frame camera-to-world poses are **stored** as a per-sensor
:class:`NumpySE3Trajectory` (``_sensor_poses``); :meth:`get_pose` returns a
single-frame slice of it and :meth:`get_poses` returns the whole trajectory, so
the SE(3) sampler can take ``boxminus`` / ``inverse`` on them directly.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

import numpy as np
import networkx as nx
from PIL import Image

from vggt_omega.datasets.sequences.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.utils.se3_trajectory import NumpySE3Trajectory


class TumSequence(BaseSequence):
    """One TUM RGB-D sequence as a :class:`BaseSequence`.

    Single sensor (``_SENSOR == "RGBD"``). Frames are RGB / depth pairs (nearest
    depth within ``rgbd_sync_diff``) carrying a groundtruth pose interpolated onto
    the color timestamp; ``frame_index`` is the 0-based position into
    that timestamp-sorted list.
    """

    _SENSOR = "RGBD"

    # Official TUM per-camera pinhole intrinsics (fx, fy, cx, cy).
    _TUM_INTRINSICS = {
        "freiburg1": (517.306408, 516.469215, 318.643040, 255.313989),
        "freiburg2": (520.908620, 521.007327, 325.141442, 249.701764),
        "freiburg3": (535.4, 539.2, 320.1, 247.6),
    }

    # -- lifecycle ----------------------------------------------------------- #
    def __init__(
        self,
        data_root: str,
        seq_id: str,
        cache_dir: Optional[str] = None,
        rgbd_sync_diff: float = 5e-3,
        depth_scale: float = 5000.0,
    ):
        """Open a TUM sequence directory.

        Args:
            data_root: directory containing the TUM sequences.
            seq_id: sequence sub-directory name (e.g.
                ``"rgbd_dataset_freiburg1_xyz"``).
            cache_dir: cache directory
            rgbd_sync_diff: max timestamp gap (s) for rgb and depth synchronization.
            depth_scale: TUM depth PNG scale (metres = pixel / depth_scale).
        """
        self._rgbd_sync_diff = float(rgbd_sync_diff)
        self._depth_scale = float(depth_scale)
        # Reference groundtruth trajectory, parsed once in load_manifest and reused
        # by load_sensor_poses to interpolate poses onto the color timestamps.
        self._gt_traj: Optional[NumpySE3Trajectory] = None
        super().__init__(data_root, seq_id, cache_dir)

    def get_sequence_directory(self) -> str:
        """Sequence directory == ``<data_root>/<seq_id>`` (no-arg override; the
        base ``__init__`` calls this with no arguments)."""
        return os.path.join(self._data_root, self._seq_id)

    def _read_tum_rgbd_txt(self, txt_file: str) -> List[Tuple[float, str]]:
        """Read TUM RGBD txt file, rgb.txt + depth.txt, [timestamp, filepath]."""
        items = []
        with open(txt_file, "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                rgbd_t, rgbd_file = line.split()
                items.append((float(rgbd_t), os.path.join(self._seq_dir, rgbd_file)))
        # sort by frame timestamps
        items = sorted(items, key=lambda x: x[0])
        return items

    def _read_tum_pose_txt(self, txt_file: str) -> np.ndarray:
        """Read TUM Groundtruth i.e. pose txt file, groundtruth.txt, [timestamp tx ty tz qx qy qz qw]"""
        poses = []
        with open(txt_file, "r") as fp:
            for line in fp.readlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                t, tx, ty, tz, qx, qy, qz, qw = [float(x) for x in line.split()]
                poses.append([t, qx, qy, qz, qw, tx, ty, tz])
        poses = np.array(poses, dtype=np.float64)
        return poses

    def load_manifest(self) -> Dict[Union[int, str], Dict[Modality, list]]:
        """Parse the rgb/depth/groundtruth indices into the per-frame manifest.

        TUM's color and depth are two separate Kinect streams logged on different
        clocks with differing frame counts, so each color frame is paired with its
        **nearest** depth frame within :attr:`_rgbd_sync_diff` (unmatched color
        frames are dropped). The groundtruth runs on yet another (mocap) clock; its
        sorted trajectory is cached on ``self._gt_traj`` and its time span gates the
        kept color frames, so the pose interpolation in :meth:`load_sensor_poses`
        never has to extrapolate. ``Modality.NAME`` is the per-frame id list (here the
        kept, timestamp-sorted positions ``0..n-1``).
        """
        rgb_items = self._read_tum_rgbd_txt(os.path.join(self._seq_dir, "rgb.txt"))
        depth_items = self._read_tum_rgbd_txt(os.path.join(self._seq_dir, "depth.txt"))

        # Reference groundtruth: sort by time (interpolate() needs an ascending
        # axis) and build the trajectory once; cache it for load_sensor_poses.
        gt_poses = self._read_tum_pose_txt(os.path.join(self._seq_dir, "groundtruth.txt"))
        gt_poses = gt_poses[np.argsort(gt_poses[:, 0])]
        self._gt_traj = NumpySE3Trajectory(
            timestamps=gt_poses[:, 0],
            poses=gt_poses[:, 1:],
            source=TumSequence._SENSOR,
            target="world",
            normalize=True,
        )
        gt_lo, gt_hi = float(gt_poses[0, 0]), float(gt_poses[-1, 0])

        depth_ts = np.array([t for t, _ in depth_items], dtype=np.float64)
        depth_paths = [p for _, p in depth_items]

        manifest = {TumSequence._SENSOR: defaultdict(list)}
        sensor = manifest[TumSequence._SENSOR]
        for rgb_t, rgb_file in rgb_items:  # rgb_items is timestamp-sorted
            if depth_ts.size == 0 or rgb_t < gt_lo or rgb_t > gt_hi:
                continue  # outside the groundtruth span -> can't interpolate a pose
            j = int(np.abs(depth_ts - rgb_t).argmin())
            if abs(depth_ts[j] - rgb_t) > self._rgbd_sync_diff:
                continue  # no depth frame close enough on the color clock
            sensor[Modality.NAME].append(len(sensor[Modality.NAME]))
            # use RGB as the RGBD sensor's reference timestamp (sync error < rgbd_sync_diff)
            sensor[Modality.TIMESTAMP].append(rgb_t)
            sensor[Modality.RGB].append(rgb_file)
            sensor[Modality.DEPTH].append(depth_paths[j])
        return manifest

    def load_calibration_tree(self) -> nx.DiGraph:
        """Single-node calibration graph carrying the constant pinhole intrinsics.

        A TUM capture is one moving RGB-D camera, so there are no inter-sensor
        extrinsics: the tree is a lone node (``_SENSOR``) whose ``"intrinsic"``
        attribute is the ``[fx, fy, cx, cy]`` pinhole intrinsic for this Freiburg
        camera (the base :meth:`get_intrinsic` / :meth:`scaled_intrinsic` read it
        from there). The camera variant (freiburg1/2/3) is inferred from ``seq_id``.
        """
        cam = next((k for k in self._TUM_INTRINSICS if k in self._seq_id), None)
        if cam is None:
            raise ValueError(
                f"no TUM intrinsics for sequence {self._seq_id!r}; "
                f"expected the id to contain one of {sorted(self._TUM_INTRINSICS)}"
            )
        fx, fy, cx, cy = self._TUM_INTRINSICS[cam]
        # The base get_intrinsic / scaled_intrinsic operate on a 4-vector
        # [fx, fy, cx, cy], so store the node intrinsic in that form.
        intrinsic = np.array([fx, fy, cx, cy], dtype=np.float32)

        tree = nx.DiGraph()
        tree.add_node(TumSequence._SENSOR, intrinsic=intrinsic)
        return tree

    def load_sensor_poses(self) -> Dict[Union[int, str], NumpySE3Trajectory]:
        """Per-frame camera-to-world poses, interpolated onto the color timestamps.

        Resamples the cached reference groundtruth trajectory (``self._gt_traj``,
        built in :meth:`load_manifest`) onto the kept color timestamps along the
        SE(3) geodesic. ``load_manifest`` already dropped frames outside the
        groundtruth span, so this never extrapolates.
        """
        rgb_ts = self._manifest[TumSequence._SENSOR][Modality.TIMESTAMP]
        rgb_traj = self._gt_traj.interpolate(rgb_ts)
        return {TumSequence._SENSOR: rgb_traj}

    # -- discovery ----------------------------------------------------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        return [TumSequence._SENSOR]

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        if sensor_id == TumSequence._SENSOR:
            return {
                Modality.RGB,
                Modality.DEPTH,
                Modality.POSE,
                Modality.TIMESTAMP,
                Modality.INTRINSIC,
                Modality.EXTRINSIC,
            }
        return set()

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._manifest[sensor_id][Modality.NAME])

    # -- per-frame getters --------------------------------------------------- #
    # Per-frame data is addressed by ``frame_index`` (0-based position into the
    # kept, timestamp-sorted timeline) -- the manifest lists and pose trajectory
    # are already in that order, so each getter indexes them directly.
    def get_rgb(self, sensor_id: Union[int, str], frame_index: int) -> np.ndarray:
        """Decode the color frame -> ``(H, W, 3)`` uint8 RGB, by position."""
        rgb_file = self._manifest[sensor_id][Modality.RGB][frame_index]
        with Image.open(rgb_file) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_rgb_semantic_mask(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no semantic masks")

    def get_rgb_dynamic_mask(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no dynamic masks")

    def get_rgb_valid_mask(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no valid masks")

    def get_depth(self, sensor_id: Union[int, str], frame_index: int) -> np.ndarray:
        """Decode the 16-bit depth PNG -> ``(H, W)`` float32 metres (0 = invalid).

        TUM stores plain uint16 integer counts; metres = value / ``depth_scale``.
        """
        depth_file = self._manifest[sensor_id][Modality.DEPTH][frame_index]
        with Image.open(depth_file) as im:
            arr = np.asarray(im).astype(np.float32)
        depth = arr / self._depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no depth confidence")

    def get_pose(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> NumpySE3Trajectory:
        """Per-frame camera-to-world pose as a **single-frame**
        :class:`NumpySE3Trajectory` (TUM groundtruth convention), by position.

        Returning a trajectory slice (rather than a pose value object) lets the
        SE(3) sampler take ``boxminus`` / ``inverse`` / ``transform_matrix`` on it
        directly."""
        return self._sensor_poses[sensor_id][frame_index]

    def get_poses(self, sensor_id: Union[int, str]) -> NumpySE3Trajectory:
        """All frame-sorted camera-to-world poses as one per-sensor
        :class:`NumpySE3Trajectory`."""
        return self._sensor_poses[sensor_id]

    # -- per-sensor / per-sequence products ---------------------------------- #
    def get_tracks(self, sensor_id: Union[int, str]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("TUM RGB-D provides no 2D/3D tracks")

    def get_pointcloud(self, sensor_id: Union[int, str]) -> np.ndarray:
        # TUM has no independent point-cloud GT (only depth re-projected through the
        # GT poses), so none is advertised here.
        raise NotImplementedError("TUM RGB-D provides no ground-truth point cloud")

    # get_pose/get_poses are overridden above; get_intrinsic / get_extrinsic are
    # inherited from BaseSequence (intrinsic from the calibration-tree node;
    # extrinsic is identity for this single-sensor capture).

    @classmethod
    def discover(
        cls, data_root: str, patterns: Optional[List[str]] = None
    ) -> List[str]:
        """TUM sequence ids = immediate sub-dirs of ``data_root`` that contain an
        ``rgb.txt`` index, filtered by the glob ``patterns`` (``["*"]`` if None)."""
        import glob

        names = set()
        for pat in patterns or ["*"]:
            for d in glob.glob(os.path.join(data_root, pat)):
                if os.path.isfile(os.path.join(d, "rgb.txt")):
                    names.add(os.path.basename(d.rstrip("/")))
        return sorted(names)

    def __repr__(self) -> str:
        return (
            f"TumSequence(seq_id={self._seq_id!r}, "
            f"frames={self.get_length(TumSequence._SENSOR)})"
        )
