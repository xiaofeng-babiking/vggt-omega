from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Union, List, Optional

if TYPE_CHECKING:
    import numpy as np

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

    # -- per-sensor / per-sequence products ---------------------------------- #
    @abstractmethod
    def get_tracks(self, sensor_id: Union[int, str]) -> tuple[np.ndarray, np.ndarray]:
        """Get 3D tracks and mask."""

    @abstractmethod
    def get_pointcloud(self) -> np.ndarray:
        """Get sparse/dense pointcloud."""

    @abstractmethod
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

        High-performance, single-sensor primitive. Returns neutral data only —
        it knows nothing about ``SequenceSample``. Multi-sensor assembly, the
        (sensor, modality) grid, ``concat`` and tensorize / GPU transfer all live
        downstream (``BaseDataset``): ``BaseSequence`` is orthogonal to the sample
        container, so nothing here imports or references it.

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
            INTRINSIC          f32  [N, 3, 3]      pinhole K, px (rescaled to image_size)
            EXTRINSIC          f32  [N, 3, 4]      w2c OpenCV (= get_extrinsic(...).transform_matrix[:3])
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
                      ``None`` -> native size (frames must already be uniform).
                      Images resize bilinear; depth / masks / confidence nearest;
                      ``INTRINSIC`` rescales to match.
        num_workers : caller-owned parallelism. ``0`` = serial — safe when nested
                      inside a torch DataLoader worker (no oversubscription). ``>0``
                      = thread pool, for a single standalone-inference parse that
                      must use the cores itself (the 141s -> 13s regime).
        image_dtype : ``"uint8"`` (transfer-light; normalize on GPU) or
                      ``"float32"`` ([0,1], normalized in-loader).

        Notes
        -----
        Deterministic decode + resize only — NO augmentation (stochastic aug is a
        separate composable transform, so ``parse`` stays reproducible / on-disk
        cacheable). No geometry derivation: ``Modality`` carries no world/cam-point
        members; unprojection from depth+pose+K happens in the loss, downstream.
        """
        # ── Design (implementation deferred) ──────────────────────────────────
        # Intended to become a CONCRETE base method (template): vendors implement
        # only the get_* getters and never override parse. Left @abstractmethod as
        # a placeholder until implemented. Sketch:
        #
        #   1. resolve modalities: (modalities or get_modalities(sensor_id)) ∩
        #      PER_FRAME; reject per-sequence / unavailable -> ValueError.
        #   2. per-frame load closure: call the matching get_* getter; resize to
        #      image_size (+ rescale K from the frame's native size); cast images
        #      per image_dtype; convert EXTRINSIC pose -> [3,4] via
        #      .transform_matrix[:3] (the one numpy<-pose boundary conversion).
        #   3. parallelism — reuse parallel_loader.py, lazy-imported so importing
        #      this ABC stays decode-free for the metadata/sampler path:
        #        · num_workers == 0 or N < _MIN_PARALLEL_FRAMES -> serial loop;
        #        · else warm-up frame_ids[0] serially (primes vendors' lazy
        #          per-sequence caches) -> ThreadPoolExecutor(num_workers) under
        #          the refcounted cv2 single-thread guard (kills N×64 thread
        #          thrash) -> fail-fast; per-call executor (fork-safe).
        #   4. merge: np.stack each modality's per-frame arrays in frame order
        #      -> [N, ...]; return dict[Modality, np.ndarray].
        # ──────────────────────────────────────────────────────────────────────
