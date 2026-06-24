from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, List, Optional, Tuple, Dict, Literal

import os
import imagesize
import numpy as np
import networkx as nx
from PIL import Image
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
    addressed by ``(sensor_id, frame_name)``. Construction reads only the manifest
    (file indices / metadata) and touches no pixels, so a sampler can query the
    discovery + pose accessors to pick frames without forcing a decode.
    ``sensor_id`` is an ``int`` index or native ``str`` key; ``frame_name`` is an
    ``int`` index into the sensor's frame-sorted timeline; the pixel getters
    decode lazily on access.
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

    def get_frame_index(self, sensor_id: Union[int, str], frame_name: int):
        """Get frame index by unique id."""
        return self._manifest[sensor_id][Modality.NAME].index(frame_name)

    def get_frame_file(
        self,
        sensor_id: Union[int, str],
        modality: Modality,
        frame_name: int,
    ):
        """Get frame file path."""
        frame_index = self.get_frame_index(sensor_id, frame_name)
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

    def get_timestamp(self, sensor_id: Union[int, str], frame_name: int) -> float:
        """Get timestamp by sensor_id and frame_name. Falls back to the sorted
        frame index when the sensor carries no ``TIMESTAMP`` modality."""
        frame_index = self.get_frame_index(sensor_id, frame_name)
        timestamps = self._manifest[sensor_id].get(Modality.TIMESTAMP)
        if not timestamps:
            return float(frame_index)
        return float(timestamps[frame_index])

    # -- per-frame getters: (sensor_id, frame_name) ---------------------------- #
    @abstractmethod
    def get_rgb(self, sensor_id: Union[int, str], frame_name: int) -> np.ndarray:
        """Get RGB image by sensor_id and frame_name."""

    def get_rgb_size(self, sensor_id: Union[int, str]) -> Tuple[int, int]:
        """Get RGB imagesize."""
        rgb_file = self._manifest[sensor_id][Modality.RGB][0]
        w, h = imagesize.get(rgb_file)
        return h, w

    @abstractmethod
    def get_rgb_semantic_mask(
        self, sensor_id: Union[int, str], frame_name: int
    ) -> np.ndarray:
        """Get per-pixel semantic-label mask, pixel-aligned to RGB."""

    @abstractmethod
    def get_rgb_dynamic_mask(
        self, sensor_id: Union[int, str], frame_name: int
    ) -> np.ndarray:
        """Get moving-object mask (True = dynamic), pixel-aligned to RGB."""

    @abstractmethod
    def get_rgb_valid_mask(
        self, sensor_id: Union[int, str], frame_name: int
    ) -> Optional[np.ndarray]:
        """Get valid mask (True = valid) to remove e.g. SKY pixels, pixel-aligned to RGB."""

    @abstractmethod
    def get_depth(self, sensor_id: Union[int, str], frame_name: int) -> np.ndarray:
        """Get depth map by sensor_id and frame_name. RGB and depth should be calibrated i.e. pixel aligned."""

    @abstractmethod
    def get_depth_confidence(
        self, sensor_id: Union[int, str], frame_name: int
    ) -> np.ndarray:
        """Get per-pixel depth confidence, aligned to depth."""

    def get_pose(self, sensor_id: Union[int, str], frame_name: int) -> np.ndarray:
        """Per-frame c2w SE(3) pose as a ``(4, 4)`` homogeneous matrix.

        ``_sensor_poses[sensor_id]`` is a :class:`BaseSE3Trajectory`; indexing it by
        frame yields a length-1 trajectory, so take its ``(1, 4, 4)``
        ``transform_matrix`` and drop the batch axis."""
        frame_index = self.get_frame_index(sensor_id, frame_name)
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

    # -- per-sensor / per-sequence products ---------------------------------- #
    @abstractmethod
    def get_tracks(self, sensor_id: Union[int, str]) -> tuple[np.ndarray, np.ndarray]:
        """Get 3D tracks and mask."""

    @abstractmethod
    def get_pointcloud(self, sensor_id: Union[int, str]) -> np.ndarray:
        """Get sparse/dense pointcloud."""

    @staticmethod
    def _resize(
        img: np.ndarray, img_size: Optional[Tuple[int, int]], interp: int
    ) -> np.ndarray:
        if img_size is None:
            return img
        h, w = img_size
        return np.asarray(Image.fromarray(img).resize((w, h), interp))

    def parse(
        self,
        sensor_id: Union[int, str],
        frame_names: List[int],
        modalities: Optional[set[Modality]] = None,
        image_size: Optional[tuple[int, int]] = None,
        image_dtype: Literal["uint8", "float32"] = "uint8",
        num_workers: int = 0,
    ) -> dict[Modality, np.ndarray]:
        """Decode ONE sensor's frames into stacked, per-modality numpy arrays.

        Concrete template: image-like modalities are decoded through the per-frame
        getters (``get_rgb`` / ``get_depth`` / ``get_rgb_semantic_mask`` /
        ``get_rgb_dynamic_mask`` / ``get_rgb_valid_mask`` / ``get_depth_confidence``),
        resized to ``image_size`` and cast per ``image_dtype``; POSE and TIMESTAMP
        are read from the precomputed ``_sensor_poses`` trajectory and the manifest.
        Each modality is stacked along axis 0 in ``frame_names`` order. Vendors
        implement only the getters + :meth:`get_modalities` + the ``load_*`` loaders;
        they need not override ``parse``.

        INTRINSIC / EXTRINSIC are **not** parsed here: they are per-sequence
        *calibration* (the sensor camera matrix; the static inter-sensor
        transform), obtained via :meth:`get_intrinsic` / :meth:`scaled_intrinsic` /
        :meth:`get_extrinsic`. Only POSE (the moving trajectory) varies per frame.

        Returns
        -------
        ``dict[Modality, np.ndarray]`` — only the requested *per-frame* modalities,
        each stacked along axis 0 (``N = len(frame_names)``; output order ==
        ``frame_names`` order)::

            RGB                u8   [N, H, W, 3]   [0,255]  (f32 [0,1] if image_dtype="float32")
            RGB_SEMANTIC_MASK  i32  [N, H, W]      label ids
            RGB_INSTANCE_MASK  i32  [N, H, W]      instance ids
            RGB_DYNAMIC_MASK   bool [N, H, W]      True = dynamic
            RGB_VALID_MASK     bool [N, H, W]      True = valid (e.g. non-sky)
            DEPTH              f32  [N, H, W]      metres · <=0 = invalid
            DEPTH_CONFIDENCE   f32  [N, H, W]
            POSE               f32  [N, 4, 4]      c2w (= get_pose(...))
            TIMESTAMP          f64  [N]            seconds
            FLOW               f32  [N, H, W, 2]   flow between consecutive frames
            TRACK              i32  [N, T, 2]      tracks
            TRACK_VISIBILITY   bool [N, T]

            POINTCLOUD         f32  [M, *]         pointcloud

        Parameters
        ----------
        sensor_id   : the sensor to decode (one synced timeline).
        frame_names   : ordered frame ids; the returned arrays follow this order.
        modalities  : subset to decode. ``None`` -> ``get_modalities(sensor_id)``
                      restricted to the per-frame group. Requesting a per-sequence
                      modality (``POINTCLOUD`` / ``TRACK``) or one the sensor does
                      not provide raises ``ValueError``.
        image_size  : ``(H, W)`` resize target so frames stack into one array.
                      ``None`` -> native size. Images resize bilinear; depth /
                      masks / confidence nearest; ``INTRINSIC`` rescales to match.
        num_workers : accepted for API compatibility; the base template decodes
                      serially (safe inside a torch DataLoader worker).
        image_dtype : ``"uint8"`` (transfer-light) or ``"float32"`` ([0,1]).
        """
        if image_dtype not in ("uint8", "float32"):
            raise ValueError(
                f"image_dtype must be 'uint8' or 'float32', got {image_dtype!r}"
            )

        available = self.get_modalities(sensor_id)
        modalities = set(available) if modalities is None else set(modalities)
        assert modalities.issubset(available), (
            f"sensor {sensor_id!r} does not provide modalities: "
            f"{sorted(m.value for m in modalities - available)}"
        )

        # Per-frame modalities decoded by looping frame_names (stacked along axis 0).
        # Each entry: (getter, PIL interp, output dtype, optional post-fn).
        frame_names = list(frame_names)
        frame_idxes = [self.get_frame_index(sensor_id, name) for name in frame_names]
        nearest, bilinear = Image.NEAREST, Image.BILINEAR

        def _get_rgb_wrapper(i):
            img = self._resize(self.get_rgb(sensor_id, i), image_size, bilinear)
            return img.astype(np.float32) / 255.0 if image_dtype == "float32" else img

        per_frame = {
            Modality.RGB: (_get_rgb_wrapper, None),
            Modality.RGB_SEMANTIC_MASK: (
                lambda i: self._resize(
                    self.get_rgb_semantic_mask(sensor_id, i), image_size, nearest
                ),
                np.int32,
            ),
            Modality.RGB_DYNAMIC_MASK: (
                lambda i: self._resize(
                    self.get_rgb_dynamic_mask(sensor_id, i), image_size, nearest
                ),
                bool,
            ),
            Modality.RGB_VALID_MASK: (
                lambda i: self._resize(
                    self.get_rgb_valid_mask(sensor_id, i), image_size, nearest
                ),
                bool,
            ),
            Modality.DEPTH: (
                lambda i: self._resize(
                    self.get_depth(sensor_id, i), image_size, nearest
                ),
                np.float32,
            ),
            Modality.DEPTH_CONFIDENCE: (
                lambda i: self._resize(
                    self.get_depth_confidence(sensor_id, i), image_size, nearest
                ),
                np.float32,
            ),
        }

        out: Dict[Modality, np.ndarray] = {}
        for mod in modalities:
            spec = per_frame.get(mod)
            if spec is None:
                continue
            getter, dtype = spec
            stacked = np.stack([np.asarray(getter(i)) for i in frame_idxes], axis=0)
            out[mod] = stacked.astype(dtype) if dtype is not None else stacked

        # TIMESTAMP: (N,) float64 vector.
        if Modality.TIMESTAMP in modalities:
            out[Modality.TIMESTAMP] = np.asarray(
                self._manifest[sensor_id][Modality.TIMESTAMP], dtype=np.float64
            )[frame_idxes]

        # POSE: (N, 4, 4) c2w, selected from the full-sequence poses.
        if Modality.POSE in modalities:
            out[Modality.POSE] = self._sensor_poses[sensor_id][
                frame_idxes
            ].transform_matrix()

        # TRACK / TRACK_VISIBILITY: per-sequence products from get_tracks (tracks, vis).
        if Modality.TRACK in modalities or Modality.TRACK_VISIBILITY in modalities:
            tracks, visibility = self.get_tracks(sensor_id)
            if Modality.TRACK in modalities:
                out[Modality.TRACK] = tracks
            if Modality.TRACK_VISIBILITY in modalities:
                out[Modality.TRACK_VISIBILITY] = visibility

        # POINTCLOUD: per-sequence (optionally per-frame) product.
        if Modality.POINTCLOUD in modalities:
            out[Modality.POINTCLOUD] = self.get_pointcloud(sensor_id)
        return out
