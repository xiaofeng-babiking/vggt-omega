from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, List, Optional, Tuple, Dict, Literal

import os
import imagesize
import numpy as np
import networkx as nx
from PIL import Image


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
    TRACK = "track"  # pixel correspondence of each track within each frame
    TRACK_VISIBILITY = "track_visibility"  # visibility of each track within each frame
    # Per-Sequence Data
    POINTCLOUD = "pointcloud"  # 3D sparse/dense pointcloud
    INTRINSIC = "intrinsic"  # camera intrinsics
    EXTRINSIC = "extrinsic"  # camera extrinsics, i.e. frame pose


class BaseSequence(ABC):
    """Abstract multi-sensor capture sequence.

    One sequence groups one or more *sensors*, each a synced timeline of frames
    addressed by ``(sensor_id, frame_id)``. Construction reads only the manifest
    (file indices / metadata) and touches no pixels, so a sampler can query the
    discovery + pose accessors to pick frames without forcing a decode.
    ``sensor_id`` / ``frame_id`` are ``int`` indices or native ``str`` keys; the
    pixel getters decode lazily on access.
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
        import os

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
        self._seq_dir = self.get_sequence_path()
        self._manifest = self.load_manifest()
        self._calib_tree = self.load_calibration_tree()
        self._sensor_poses = self.load_sensor_poses()

    def get_sequence_directory(self, data_root: str, seq_id: str) -> str:
        """Return sequence directory by data root path and sequence id."""
        return os.path.join(data_root, seq_id)

    @abstractmethod
    def load_manifest(self) -> Dict[Union[int, str], Dict[Modality, str]]:
        """Load sequence manifest, e.g. global indices of files."""

    @abstractmethod
    def load_calibration_tree(self) -> nx.DiGraph:
        """Load extrinsics calibration tree, i.e. relative pose between sensors."""

    @abstractmethod
    def load_sensor_poses(self) -> Dict[Union[int, str], np.ndarray]:
        """ "Load frame poses for each sensor."""

    def get_frame_file(
        self,
        sensor_id: Union[int, str],
        modality: Modality,
        frame_id: int,
    ):
        """Get frame file path."""
        return self._manifest[sensor_id][modality][frame_id]

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

    @abstractmethod
    def get_timestamp(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> float:
        """Get timestamps by sensor_id and frame_id. If no frame timestamp, use sorted frame index instead."""

    # -- per-frame getters: (sensor_id, frame_id) ---------------------------- #
    @abstractmethod
    def get_rgb(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Get RGB image by sensor_id and frame_id."""

    @abstractmethod
    def get_rgb_size(self, sensor_id: Union[int, str]) -> Tuple[int, int]:
        """Get RGB imagesize."""
        rgb_file = self._manifest[sensor_id][Modality.RGB][0]
        w, h = imagesize.get(rgb_file)
        return h, w

    @abstractmethod
    def get_rgb_semantic_mask(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Get per-pixel semantic-label mask, pixel-aligned to RGB."""

    @abstractmethod
    def get_rgb_dynamic_mask(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Get moving-object mask (True = dynamic), pixel-aligned to RGB."""

    @abstractmethod
    def get_rgb_valid_mask(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> Optional[np.ndarray]:
        """Get valid mask (True = valid) to remove e.g. SKY pixels, pixel-aligned to RGB."""

    @abstractmethod
    def get_depth(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Get depth map by sensor_id and frame_id. RGB and depth should be calibrated i.e. pixel aligned."""

    @abstractmethod
    def get_depth_confidence(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Get per-pixel depth confidence, aligned to depth."""

    def get_pose(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Per-frame c2w SE(3) pose as a ``(4, 4)`` homogeneous matrix."""
        return self._sensor_poses[sensor_id][frame_id]

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
        """Get intrinsic camera matrix ``(3, 3)`` for ``sensor_id`` from the
        calibration tree node attribute ``"intrinsic"``."""
        return self._calib_tree.nodes[sensor_id]["intrinsic"]

    def scaled_intrinsic(
        self,
        sensor_id: Union[int, str],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """``(3, 3)`` per-sequence intrinsic rescaled to ``image_size`` ``(H, W)``.

        Concrete in the base: pulls the sensor's calibrated ``get_intrinsic`` and,
        when ``image_size`` is given, rescales ``fx, cx`` by ``W/W0`` and
        ``fy, cy`` by ``H/H0``, where the native ``(H0, W0)`` is read lazily from
        the sensor's first frame image header. ``image_size=None`` returns the
        native intrinsic unchanged. (Intrinsics are per-sequence calibration, so
        this keys off ``sensor_id`` only — not a frame.)"""
        K = np.asarray(self.get_intrinsic(sensor_id), dtype=np.float32).copy()
        if image_size is None:
            return K
        h0, w0 = self.get_rgb_size(sensor_id)
        sy = image_size[0] / float(h0)
        sx = image_size[1] / float(w0)
        K[0, 0] *= sx  # fx
        K[0, 2] *= sx  # cx
        K[1, 1] *= sy  # fy
        K[1, 2] *= sy  # cy
        return K

    # -- per-sensor / per-sequence products ---------------------------------- #
    @abstractmethod
    def get_tracks(self, sensor_id: Union[int, str]) -> tuple[np.ndarray, np.ndarray]:
        """Get 3D tracks and mask."""

    @abstractmethod
    def get_pointcloud(
        self, sensor_id: Union[int, str], frame_ids: Optional[List[int]] = None
    ) -> np.ndarray:
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
        frame_ids: List[Union[int, str]],
        modalities: Optional[set[Modality]] = None,
        image_size: Optional[tuple[int, int]] = None,
        image_dtype: Literal["uint8", "float32"] = "uint8",
        num_workers: int = 0,
    ) -> dict[Modality, np.ndarray]:
        """Decode ONE sensor's frames into stacked, per-modality numpy arrays.

        Concrete template: composes the per-frame getters (``get_rgb`` /
        ``get_depth`` / ``get_rgb_semantic_mask`` / ``get_rgb_dynamic_mask`` /
        ``get_rgb_valid_mask`` / ``get_depth_confidence`` / ``get_pose`` /
        ``get_timestamp``), resizes to ``image_size``, casts images per
        ``image_dtype`` and stacks each modality along axis 0 in ``frame_ids``
        order. Vendors implement only the getters + :meth:`get_modalities` +
        :meth:`_frame_image_path`; they need not override ``parse``.

        INTRINSIC / EXTRINSIC are **not** parsed here: they are per-sequence
        *calibration* (the sensor camera matrix; the static inter-sensor
        transform), obtained via :meth:`get_intrinsic` / :meth:`scaled_intrinsic` /
        :meth:`get_extrinsic`. Only POSE (the moving trajectory) varies per frame.

        Returns
        -------
        ``dict[Modality, np.ndarray]`` — only the requested *per-frame* modalities,
        each stacked along axis 0 (``N = len(frame_ids)``; output order ==
        ``frame_ids`` order)::

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
        frame_ids   : ordered frame ids; the returned arrays follow this order.
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
        requested = set(modalities) if modalities is not None else set(available)
        unavailable = requested - available
        if unavailable:
            raise ValueError(
                f"sensor {sensor_id!r} does not provide modalities: "
                f"{sorted(m.value for m in unavailable)}"
            )

        # Per-frame modalities decoded by looping frame_ids (stacked along axis 0).
        # Each entry: (getter, PIL interp, output dtype, optional post-fn).
        frame_ids = list(frame_ids)
        nearest, bilinear = Image.NEAREST, Image.BILINEAR

        def _rgb(i):
            img = self._resize(self.get_rgb(sensor_id, i), image_size, bilinear)
            return img.astype(np.float32) / 255.0 if image_dtype == "float32" else img

        per_frame = {
            Modality.RGB: (_rgb, None),
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
        for m in requested:
            spec = per_frame.get(m)
            if spec is None:
                continue
            getter, dtype = spec
            stacked = np.stack([np.asarray(getter(i)) for i in frame_ids], axis=0)
            out[m] = stacked.astype(dtype) if dtype is not None else stacked

        # TIMESTAMP: (N,) float64 vector.
        if Modality.TIMESTAMP in requested:
            out[Modality.TIMESTAMP] = np.asarray(
                [self.get_timestamp(sensor_id, i) for i in frame_ids], dtype=np.float64
            )

        # POSE: (N, 4, 4) c2w, selected from the (cached) full-sequence poses.
        if Modality.POSE in requested:
            poses = np.asarray(self.get_poses(sensor_id), dtype=np.float32)
            out[Modality.POSE] = poses[np.asarray(frame_ids, dtype=int)]

        # TRACK / TRACK_VISIBILITY: per-sequence products from get_tracks (tracks, vis).
        if Modality.TRACK in requested or Modality.TRACK_VISIBILITY in requested:
            tracks, visibility = self.get_tracks(sensor_id)
            if Modality.TRACK in requested:
                out[Modality.TRACK] = np.asarray(tracks)
            if Modality.TRACK_VISIBILITY in requested:
                out[Modality.TRACK_VISIBILITY] = np.asarray(visibility)

        # POINTCLOUD: per-sequence (optionally per-frame) product.
        if Modality.POINTCLOUD in requested:
            out[Modality.POINTCLOUD] = self.get_pointcloud(sensor_id, frame_ids)

        return out
