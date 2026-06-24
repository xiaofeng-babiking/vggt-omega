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
:class:`NumpySE3Trajectory` (``_sensor_poses``) and **returned** as
:class:`NumpySE3Pose` value objects from :meth:`get_pose` / :meth:`get_poses` (so
the SE(3) sampler can take ``boxminus`` / ``inverse`` on them).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict

import numpy as np
import networkx as nx
from PIL import Image

from vggt_omega.datasets.sequences.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.se3_pose import BaseSE3Pose, NumpySE3Pose
from vggt_omega.datasets.utils.se3_trajectory import NumpySE3Trajectory


class TumSequence(BaseSequence):
    """One TUM RGB-D sequence as a :class:`BaseSequence`.

    Single sensor (``_SENSOR == "RGBD"``). Frames are RGB / depth pairs (nearest
    depth within ``rgbd_sync_diff``) carrying a groundtruth pose interpolated onto
    the color timestamp; ``frame_id`` is the integer index into that
    timestamp-sorted list.
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
        """Parse the rgb/depth/groundtruth indices into the per-frame manifest."""
        rgb_items = self._read_tum_rgbd_txt(os.path.join(self._seq_dir, "rgb.txt"))
        depth_items = self._read_tum_rgbd_txt(os.path.join(self._seq_dir, "depth.txt"))

        manifest = {TumSequence._SENSOR: defaultdict(list)}
        sensor = manifest[TumSequence._SENSOR]
        for (rgb_t, rgb_file), (depth_t, depth_file) in zip(rgb_items, depth_items):
            assert abs(rgb_t - depth_t) <= self._rgbd_sync_diff

            sensor[Modality.NAME].append(len(sensor[Modality.NAME]))
            # use RGB as the RGBD sensor's reference timestamp (sync error < rgbd_sync_diff)
            sensor[Modality.TIMESTAMP].append(rgb_t)
            sensor[Modality.RGB].append(rgb_file)
            sensor[Modality.DEPTH].append(depth_file)
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
        """Load sensor poses."""
        gt_poses = self._read_tum_pose_txt(
            os.path.join(self._seq_dir, "groundtruth.txt")
        )
        # interpolate() assumes an ascending time axis (searchsorted + endpoint
        # range check), so sort the groundtruth rows by timestamp first.
        gt_poses = gt_poses[np.argsort(gt_poses[:, 0])]

        gt_traj = NumpySE3Trajectory(
            timestamps=gt_poses[:, 0],
            poses=gt_poses[:, 1:],
            source=TumSequence._SENSOR,
            target="world",
            normalize=True,
        )

        rgb_ts = self._manifest[TumSequence._SENSOR][Modality.TIMESTAMP]
        rgb_traj = gt_traj.interpolate(rgb_ts)

        sensor_poses = {TumSequence._SENSOR: rgb_traj}
        return sensor_poses

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

    def get_timestamp(self, sensor_id: Union[int, str], frame_id: int = 0) -> float:
        return float(self._manifest[sensor_id][Modality.TIMESTAMP][frame_id])

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(
        self,
        sensor_id: Union[int, str],
        frame_id: int,
    ) -> np.ndarray:
        """Decode the color frame -> ``(H, W, 3)`` uint8 RGB."""
        rgb_file = self._manifest[sensor_id][Modality.RGB][frame_id]
        with Image.open(rgb_file) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_rgb_semantic_mask(
        self, sensor_id: Union[int, str], frame_id: int
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no semantic masks")

    def get_rgb_dynamic_mask(
        self, sensor_id: Union[int, str], frame_id: int
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no dynamic masks")

    def get_rgb_valid_mask(
        self, sensor_id: Union[int, str], frame_id: int
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no valid masks")

    def get_depth(self, sensor_id: Union[int, str], frame_id: int) -> np.ndarray:
        """Decode the 16-bit depth PNG -> ``(H, W)`` float32 metres (0 = invalid).

        TUM stores plain uint16 integer counts; metres = value / ``depth_scale``.
        """
        depth_file = self._manifest[sensor_id][Modality.DEPTH][frame_id]
        with Image.open(depth_file) as im:
            arr = np.asarray(im).astype(np.float32)
        depth = arr / self._depth_scale
        depth[~np.isfinite(depth)] = 0.0
        return depth

    def get_depth_confidence(
        self, sensor_id: Union[int, str], frame_id: int
    ) -> np.ndarray:
        raise NotImplementedError("TUM RGB-D provides no depth confidence")

    def get_pose(self, sensor_id: Union[int, str], frame_id: int) -> np.ndarray:
        """Per-frame camera-to-world SE(3) pose (TUM groundtruth convention).

        Wraps the stored ``(4, 4)`` c2w matrix into a :class:`NumpySE3Pose` value
        object (the form the SE(3) sampler / dataset adapter consume)."""
        tf_c2w = self._sensor_poses[sensor_id][int(frame_id)].transform_matrix()[0]
        return NumpySE3Pose.from_tf_mat(tf_c2w)

    def get_poses(self, sensor_id: Union[int, str]) -> List[BaseSE3Pose]:
        """All frame-sorted camera-to-world poses as :class:`NumpySE3Pose` objects."""
        return [self.get_pose(sensor_id, i) for i in range(self.get_length(sensor_id))]

    # -- per-sensor / per-sequence products ---------------------------------- #
    def get_tracks(self, sensor_id: Union[int, str]) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("TUM RGB-D provides no 2D/3D tracks")

    def get_pointcloud(
        self, sensor_id: Union[int, str], frame_ids: Optional[List[int]] = None
    ) -> np.ndarray:
        # TUM has no independent point-cloud GT (only depth re-projected through the
        # GT poses), so none is advertised here.
        raise NotImplementedError("TUM RGB-D provides no ground-truth point cloud")

    # get_pose/get_poses are overridden above; get_intrinsic / get_extrinsic are
    # inherited from BaseSequence (intrinsic from the calibration-tree node;
    # extrinsic is identity for this single-sensor capture). parse() is inherited
    # as the concrete template over the getters.

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
