from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Union, List, Optional, Tuple

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .se3_pose import BaseSE3Pose


class Modality(str, Enum):
    """Data Modality Enumerate."""

    # Per-Frame Data
    RGB = "rgb"  # RGB image
    RGB_SEMANTIC_MASK = "rgb_semantic_mask"
    RGB_DYNAMIC_MASK = "rgb_dynamic_mask"
    DEPTH = "depth"  # depth image
    DEPTH_CONFIDENCE = "depth_confidence"  # 2D pixel confidence
    POSE = "pose"  # frame pose
    TIMESTAMP = "timestamp"
    # Per-Sequence Data
    POINTCLOUD = "pointcloud"  # 3D sparse/dense pointcloud
    TRACK = "track"  # 2D pixel tracking across frames
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

    # Modalities ``parse`` stacks per frame. INTRINSIC / EXTRINSIC are NOT here:
    # they are per-sequence *calibration* (a sensor's camera matrix; the static
    # inter-sensor transform), fetched via get_intrinsic(sensor_id) /
    # get_extrinsic(src, dst) — not per-frame. POSE (the moving trajectory) is the
    # only geometry that varies per frame. POINTCLOUD / TRACK are per-sequence.
    _PER_FRAME_MODALITIES = frozenset(
        {
            Modality.RGB,
            Modality.RGB_SEMANTIC_MASK,
            Modality.RGB_DYNAMIC_MASK,
            Modality.DEPTH,
            Modality.DEPTH_CONFIDENCE,
            Modality.POSE,
            Modality.TIMESTAMP,
        }
    )

    # -- discovery (which sequence ids live under a data root) --------------- #
    @classmethod
    def discover(cls, data_root: str, patterns: "Optional[List[str]]" = None) -> "List[str]":
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
        for pat in (patterns or ["*"]):
            for d in glob.glob(os.path.join(data_root, pat)):
                if os.path.isdir(d):
                    names.add(os.path.relpath(d, data_root))
        return sorted(names)

    # -- lifecycle ----------------------------------------------------------- #
    @abstractmethod
    def __init__(self, data_root: str, seq_id: str):
        """Initialize sequence."""
    @abstractmethod
    def load_manifest(self) -> None:
        """Load sequence manifest, e.g. global indices of files."""

    @abstractmethod
    def load_extrinsics(self):
        """Load extrinsics calibration, i.e. relative pose between sensors."""

    @abstractmethod
    def load_intrinsics(self):
        """Load sensor intrinsics, e.g. camera matrix."""

    # -- discovery (cheap metadata; never triggers a decode) ----------------- #
    @abstractmethod
    def get_sensors(self) -> list[Union[int, str]]:
        """List available sensor ids. Each sensor is one synced frame timeline."""

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
    def get_semantic_mask(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Get per-pixel semantic-label mask, pixel-aligned to RGB."""

    @abstractmethod
    def get_dynamic_mask(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> np.ndarray:
        """Get moving-object mask (True = dynamic), pixel-aligned to RGB."""

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

    @abstractmethod
    def get_pose(self, sensor_id: Union[int, str], frame_id: Union[int, str]):
        """Get per-frame SE3 pose."""

    @abstractmethod
    def get_extrinsic(
        self, src_sensor_id: Union[int, str], dst_sensor_id: Union[int, str]
    ) -> BaseSE3Pose:
        """Get extrinsic i.e. static SE3 transform from source to target sensor id."""

    @abstractmethod
    def get_intrinsic(
        self,
        sensor_id: Union[int, str],
    ) -> np.ndarray:
        """Get intrinsic, e.g. camera matrix."""

    # -- image-size helpers (manifest-backed; concrete in the base) ---------- #
    @abstractmethod
    def _frame_image_path(
        self, sensor_id: Union[int, str], frame_id: Union[int, str]
    ) -> str:
        """Path to the (undecoded) RGB file for a frame.

        The base owns the generic header read / intrinsic rescale; only this
        manifest-specific lookup is per-vendor, since the manifest schema (where
        file paths live) is private to each subclass."""

    @staticmethod
    def read_image_size(path: str) -> Tuple[int, int]:
        """Read an image's ``(H, W)`` from its header — no pixel decode.

        Generic primitive shared by all vendors; keeps the metadata / sampler path
        decode-free (PIL reads dimensions without decompressing the image)."""
        with Image.open(path) as im:
            w, h = im.size  # PIL reports (W, H)
        return (h, w)

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
        h0, w0 = self.read_image_size(self._frame_image_path(sensor_id, 0))
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
    def get_pointcloud(self) -> np.ndarray:
        """Get sparse/dense pointcloud."""

    def parse(
        self,
        sensor_id: Union[int, str],
        frame_ids: List[Union[int, str]],
        modalities: Optional[set[Modality]] = None,
        image_size: Optional[tuple[int, int]] = None,
        image_dtype: str = "uint8",
        num_workers: int = 0,
    ) -> dict[Modality, np.ndarray]:
        """Decode ONE sensor's frames into stacked, per-modality numpy arrays.

        Concrete template: composes the per-frame getters (``get_rgb`` /
        ``get_depth`` / ``get_semantic_mask`` / ``get_dynamic_mask`` /
        ``get_depth_confidence`` / ``get_pose`` / ``get_timestamp``), resizes to
        ``image_size``, casts images per ``image_dtype`` and stacks each modality
        along axis 0 in ``frame_ids`` order. Vendors implement only the getters +
        :meth:`get_modalities` + :meth:`_frame_image_path`; they need not override
        ``parse``.

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
            RGB_DYNAMIC_MASK   bool [N, H, W]      True = dynamic
            DEPTH              f32  [N, H, W]      metres · 0 = invalid · <0 = sky
            DEPTH_CONFIDENCE   f32  [N, H, W]
            POSE               f32  [N, 4, 4]      c2w (= get_pose(...).transform_matrix)
            TIMESTAMP          f64  [N]            seconds

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
            raise ValueError(f"image_dtype must be 'uint8' or 'float32', got {image_dtype!r}")

        available = self.get_modalities(sensor_id)
        per_frame = available & self._PER_FRAME_MODALITIES
        requested = set(modalities) if modalities is not None else set(per_frame)

        non_per_frame = requested - self._PER_FRAME_MODALITIES
        if non_per_frame:
            raise ValueError(
                f"parse handles per-frame modalities only; got "
                f"{sorted(m.value for m in non_per_frame)}"
            )
        unavailable = requested - available
        if unavailable:
            raise ValueError(
                f"sensor {sensor_id!r} does not provide modalities: "
                f"{sorted(m.value for m in unavailable)}"
            )

        def _resize(arr: "np.ndarray", interp: int) -> "np.ndarray":
            if image_size is None:
                return arr
            return np.asarray(
                Image.fromarray(arr).resize((image_size[1], image_size[0]), interp)
            )

        cols: dict[Modality, list] = {m: [] for m in requested}
        for f in frame_ids:
            if Modality.RGB in requested:
                rgb = _resize(self.get_rgb(sensor_id, f), Image.BILINEAR)
                if image_dtype == "float32":
                    rgb = rgb.astype(np.float32) / 255.0
                cols[Modality.RGB].append(rgb)
            if Modality.RGB_SEMANTIC_MASK in requested:
                cols[Modality.RGB_SEMANTIC_MASK].append(
                    _resize(self.get_semantic_mask(sensor_id, f), Image.NEAREST)
                )
            if Modality.RGB_DYNAMIC_MASK in requested:
                cols[Modality.RGB_DYNAMIC_MASK].append(
                    _resize(self.get_dynamic_mask(sensor_id, f), Image.NEAREST)
                )
            if Modality.DEPTH in requested:
                cols[Modality.DEPTH].append(
                    _resize(self.get_depth(sensor_id, f), Image.NEAREST)
                )
            if Modality.DEPTH_CONFIDENCE in requested:
                cols[Modality.DEPTH_CONFIDENCE].append(
                    _resize(self.get_depth_confidence(sensor_id, f), Image.NEAREST)
                )
            if Modality.POSE in requested:
                cols[Modality.POSE].append(
                    np.asarray(self.get_pose(sensor_id, f).transform_matrix, dtype=np.float32)
                )
            if Modality.TIMESTAMP in requested:
                cols[Modality.TIMESTAMP].append(self.get_timestamp(sensor_id, f))

        out: dict[Modality, np.ndarray] = {}
        for m, vals in cols.items():
            if m is Modality.TIMESTAMP:
                out[m] = np.asarray(vals, dtype=np.float64)
            else:
                out[m] = np.stack(vals, axis=0)
        return out
        # ──────────────────────────────────────────────────────────────────────
