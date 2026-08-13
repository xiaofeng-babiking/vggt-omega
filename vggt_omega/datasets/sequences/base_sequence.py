from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, List, Optional, Tuple, Dict

import os
import imagesize
import numpy as np
import networkx as nx
from vggt_omega.datasets.utils.se3_trajectory import BaseSE3Trajectory


class Modality(str, Enum):
    """Data Modality Enumerate."""

    # Per-Frame Data
    RGB = "rgb"  # RGB image
    RGB_SEMANTIC_MASK = "rgb_semantic_mask"
    RGB_INSTANCE_MASK = "rgb_instance_mask"
    RGB_DYNAMIC_MASK = "rgb_dynamic_mask"
    RGB_VALID_MASK = "rgb_valid_mask"
    DEPTH = "depth"  # depth image
    DEPTH_CONFIDENCE = "depth_confidence"  # 2D pixel confidence
    FLOW = "flow"  # flow map
    POSE = "pose"  # frame pose
    TIMESTAMP = "timestamp"
    NAME = "name"  # frame name i.e. unique id for each frame
    TRACK = "track"  # pixel correspondence of each track within each frame
    TRACK_VISIBILITY = "track_visibility"  # visibility of each track within each frame
    # Per-Sequence Data
    POINTCLOUD = "pointcloud"  # 3D sparse/dense pointcloud
    INTRINSIC = "intrinsic"  # camera intrinsics
    EXTRINSIC = "extrinsic"  # camera extrinsics, i.e. frame pose


class BaseSequence(ABC):
    """Abstract multi-sensor capture sequence.

    One sequence groups one or more *sensors*, each a synced timeline of frames
    addressed by ``(sensor_id, frame_index)``, where ``frame_index`` is the 0-based
    position into the sensor's frame-sorted timeline. Construction reads only the
    manifest (file indices / metadata) and touches no pixels, so a sampler can
    query the discovery + pose accessors to pick frames without forcing a decode.
    ``sensor_id`` is an ``int`` index or native ``str`` key; the pixel getters
    decode lazily on access.

    Frames also carry an arbitrary, stable *name* (``Modality.NAME``; an ``int``,
    ``str`` or other hashable id). Names are an optional user-facing namespace:
    :meth:`get_frame_index` / :meth:`get_frame_indices` map a name to its position.
    The decode pipeline addresses frames by position only and never resolves names.
    """

    # -- discovery (which sequence ids live under a data root) --------------- #
    @classmethod
    def discover(
        cls, data_root: str, patterns: "Optional[List[str]]" = None
    ) -> "List[str]":
        """List the sequence ids under ``data_root`` (relative ids passable as
        ``seq_id`` to this class).

        Default: each immediate sub-directory of ``data_root`` matching a glob in
        ``patterns`` (``["*"]`` if None) is one sequence. Vendors whose sequences
        are nested, require a marker file, or come from an index file override
        this (the layout knowledge belongs with the BaseSequence subclass, so a
        single generic ``SequenceDataset`` can stay vendor-agnostic).
        """
        import glob

        names = set()
        for pat in patterns or ["*"]:
            for d in glob.glob(os.path.join(data_root, pat)):
                if os.path.isdir(d):
                    names.add(os.path.relpath(d, data_root))
        return sorted(names)

    # -- lifecycle ----------------------------------------------------------- #
    def __init__(
        self,
        data_root: str,
        seq_id: str,
        cache_dir: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Initialize sequence."""
        self._data_root = data_root
        self._seq_id = seq_id
        self._cache_dir = cache_dir
        self._seq_dir = self.get_sequence_directory()
        self._manifest = self.load_manifest()
        self._calib_tree = self.load_calibration_tree()
        self._sensor_poses = self.load_sensor_poses()

    def get_sequence_directory(self) -> str:
        """Return sequence directory ``<data_root>/<seq_id>`` (called arg-free from
        :meth:`__init__`; reads ``self._data_root`` / ``self._seq_id``)."""
        return os.path.join(self._data_root, self._seq_id)

    @abstractmethod
    def load_manifest(self) -> Dict[Union[int, str], Dict[Modality, str]]:
        """Load sequence manifest, e.g. global indices of files."""

    @abstractmethod
    def load_calibration_tree(self) -> nx.DiGraph:
        """Load extrinsics calibration tree, i.e. relative pose between sensors."""

    @abstractmethod
    def load_sensor_poses(self) -> Dict[Union[int, str], BaseSE3Trajectory]:
        """Load frame poses for each sensor. Pose format: [qx, qy, qz, qw, tx, ty, tz]."""

    def get_frame_index(
        self, sensor_id: Union[int, str], frame_name: Union[int, str]
    ) -> int:
        """Map an arbitrary frame name to its 0-based position (user utility; the
        decode pipeline addresses frames by position and never calls this)."""
        return self._manifest[sensor_id][Modality.NAME].index(frame_name)

    def get_frame_indices(
        self, sensor_id: Union[int, str], frame_names: List[Union[int, str]]
    ) -> List[int]:
        """Get frame indices by unique ids (bulk find/map).

        High-performance counterpart of :meth:`get_frame_index`: builds a
        ``name -> position`` map over the sensor's frame-sorted ``NAME`` list
        once (``O(M)``) and resolves each requested name in ``O(1)``, i.e.
        ``O(M + N)`` total versus the ``O(N * M)`` of calling ``list.index`` per
        name. Frame names are unique ids (``Modality.NAME``), so the map is
        unambiguous. Returned indices follow ``frame_names`` order; a name absent
        from the sensor's timeline raises ``KeyError``.
        """
        names = self._manifest[sensor_id][Modality.NAME]
        name_to_index = {name: index for index, name in enumerate(names)}
        return [name_to_index[name] for name in frame_names]

    def get_frame_file(
        self,
        sensor_id: Union[int, str],
        modality: Modality,
        frame_index: int,
    ):
        """Get frame file path by position."""
        return self._manifest[sensor_id][modality][frame_index]

    # -- discovery (cheap metadata; never triggers a decode) ----------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        """List available sensor ids. Each sensor is one synced frame timeline."""
        return list(self._manifest.keys())

    @abstractmethod
    def get_modalities(self, sensor_id: Union[int, str]) -> set[Modality]:
        """Return the set of Modality a sensor provides (e.g. a lidar lacks RGB)."""

    @abstractmethod
    def get_length(self, sensor_id: Union[int, str]) -> int:
        """Return number of frames of specific sensor."""

    def get_timestamp(self, sensor_id: Union[int, str], frame_index: int) -> float:
        """Get timestamp by sensor_id and frame position. Falls back to the frame
        index itself when the sensor carries no ``TIMESTAMP`` modality."""
        timestamps = self._manifest[sensor_id].get(Modality.TIMESTAMP)
        if not timestamps:
            return float(frame_index)
        return float(timestamps[frame_index])

    # -- per-frame getters: (sensor_id, frame_index) --------------------------- #
    @abstractmethod
    def get_rgb(self, sensor_id: Union[int, str], frame_index: int) -> np.ndarray:
        """Get RGB image by sensor_id and frame position."""

    def get_rgb_size(self, sensor_id: Union[int, str]) -> Tuple[int, int]:
        """Get RGB imagesize."""
        rgb_file = self._manifest[sensor_id][Modality.RGB][0]
        w, h = imagesize.get(rgb_file)
        return h, w

    @abstractmethod
    def get_rgb_semantic_mask(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> np.ndarray:
        """Get per-pixel semantic-label mask, pixel-aligned to RGB."""

    @abstractmethod
    def get_rgb_dynamic_mask(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> np.ndarray:
        """Get moving-object mask (True = dynamic), pixel-aligned to RGB."""

    @abstractmethod
    def get_rgb_valid_mask(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> Optional[np.ndarray]:
        """Get valid mask (True = valid) to remove e.g. SKY pixels, pixel-aligned to RGB."""

    @abstractmethod
    def get_depth(self, sensor_id: Union[int, str], frame_index: int) -> np.ndarray:
        """Get depth map by sensor_id and frame position. RGB and depth should be calibrated i.e. pixel aligned."""

    @abstractmethod
    def get_depth_confidence(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> np.ndarray:
        """Get per-pixel depth confidence, aligned to depth."""

    def get_pose(self, sensor_id: Union[int, str], frame_index: int) -> np.ndarray:
        """Per-frame c2w SE(3) pose as a ``(4, 4)`` homogeneous matrix, by position.

        ``_sensor_poses[sensor_id]`` is a :class:`BaseSE3Trajectory`; indexing it by
        position yields a length-1 trajectory, so take its ``(1, 4, 4)``
        ``transform_matrix`` and drop the batch axis."""
        return self._sensor_poses[sensor_id][frame_index].transform_matrix()[0]

    def get_poses(self, sensor_id: Union[int, str]) -> np.ndarray:
        """Frame c2w SE(3) poses for a sensor as ``(M, 4, 4)``, frame-sorted
        (precomputed in :meth:`load_sensor_poses`)."""
        return self._sensor_poses[sensor_id].transform_matrix()

    def get_extrinsic(
        self, src_sensor_id: Union[int, str], dst_sensor_id: Union[int, str]
    ) -> np.ndarray:
        """Static ``(4, 4)`` SE(3) transform from ``src_sensor_id`` frame to
        ``dst_sensor_id`` frame, composed along the calibration tree.

        ``self._calib_tree`` is an ``nx.DiGraph`` whose edges carry an
        ``"extrinsic"`` attribute (a ``(4, 4)`` homogeneous matrix): a directed edge
        ``u -> v`` stores the transform that maps points from frame ``u`` into frame
        ``v``. We take the shortest path ``src -> ... -> dst`` over the *undirected*
        view (so a stored ``u -> v`` edge can be traversed backwards as its inverse)
        and compose the per-edge matrices in order.

        ``src == dst`` returns identity. Raises ``networkx.NetworkXNoPath`` (or
        ``NodeNotFound``) if the two sensors are not connected in the tree.
        """
        if src_sensor_id == dst_sensor_id:
            return np.eye(4, dtype=np.float64)

        # Shortest hop path over the undirected view; each hop is a stored edge in
        # one direction or the inverse of the reverse edge.
        path = nx.shortest_path(
            self._calib_tree.to_undirected(as_view=True), src_sensor_id, dst_sensor_id
        )
        result = np.eye(4, dtype=np.float64)
        for u, v in zip(path[:-1], path[1:]):
            if self._calib_tree.has_edge(u, v):
                step = np.asarray(
                    self._calib_tree.edges[u, v]["extrinsic"], dtype=np.float64
                )
            else:
                step = np.linalg.inv(
                    np.asarray(
                        self._calib_tree.edges[v, u]["extrinsic"], dtype=np.float64
                    )
                )
            # Accumulate dst<-src: apply earlier hops first, then this one.
            result = step @ result
        return result

    def get_intrinsic(
        self,
        sensor_id: Union[int, str],
    ) -> np.ndarray:
        """Get intrinsic camera matrix ``[4] -> [fx, fy, cx, cy]`` for ``sensor_id`` from the
        calibration tree node attribute ``"intrinsic"``."""
        return self._calib_tree.nodes[sensor_id]["intrinsic"]

    def scaled_intrinsic(
        self,
        sensor_id: Union[int, str],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """``[4] -> [fx, fy, cx, cy]`` per-sequence intrinsic rescaled to ``image_size`` ``(H, W)``.

        Concrete in the base: pulls the sensor's calibrated ``get_intrinsic`` and,
        when ``image_size`` is given, rescales ``fx, cx`` by ``W/W0`` and
        ``fy, cy`` by ``H/H0``, where the native ``(H0, W0)`` is read lazily from
        the sensor's first frame image header. ``image_size=None`` returns the
        native intrinsic unchanged. (Intrinsics are per-sequence calibration, so
        this keys off ``sensor_id`` only — not a frame.)"""
        intrinsic = np.asarray(self.get_intrinsic(sensor_id), dtype=np.float32).copy()
        if image_size is None:
            return intrinsic
        h0, w0 = self.get_rgb_size(sensor_id)
        sy = image_size[0] / float(h0)
        sx = image_size[1] / float(w0)
        intrinsic[0] *= sx  # fx
        intrinsic[1] *= sy  # fy
        intrinsic[2] *= sx  # cx
        intrinsic[3] *= sy  # cy
        return intrinsic

    @staticmethod
    def rescale_intrinsic(
        intrinsic: np.ndarray,
        src_size_wh: Tuple[int, int],
        dst_size_wh: Tuple[int, int],
    ) -> np.ndarray:
        """Rescale ``[fx, fy, cx, cy]`` from ``src_size_wh`` to ``dst_size_wh`` (both ``(W, H)``).

        Pure per-axis scaling: ``fx, cx`` scale with width, ``fy, cy`` with height.
        A ``@staticmethod`` (no instance state) so it can run *during*
        calibration-tree construction — e.g. rescaling a COLMAP intrinsic onto a
        downsampled ``images_2/4/8`` resolution — where ``scaled_intrinsic`` (which
        reads the not-yet-built tree) can't be called. Callers pass both sizes
        explicitly, e.g. ``self.rescale_intrinsic(K, (w0, h0), (w, h))``.
        """
        fx, fy, cx, cy = np.asarray(intrinsic, dtype=np.float64)
        (w0, h0), (w, h) = src_size_wh, dst_size_wh
        sx, sy = w / float(w0), h / float(h0)
        return np.array([fx * sx, fy * sy, cx * sx, cy * sy], dtype=np.float64)

    # -- per-sensor / per-sequence products ---------------------------------- #
    @abstractmethod
    def get_tracks(self, sensor_id: Union[int, str]) -> tuple[np.ndarray, np.ndarray]:
        """Get 3D tracks and mask."""

    @abstractmethod
    def get_pointcloud(self, sensor_id: Union[int, str]) -> np.ndarray:
        """Get sparse/dense pointcloud."""
