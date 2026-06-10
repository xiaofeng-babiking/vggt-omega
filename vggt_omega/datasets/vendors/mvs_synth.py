"""MVS-Synth vendor for the VGGT-Omega dataset API.

MVS-Synth is a photorealistic synthetic dataset rendered from GTA-V. This is a
preprocessed copy (the original EXR depth + per-frame camera JSON were converted
to ``.npy`` / ``.npz``) with a single ``train`` split of 120 sequences x 100
frames each::

    {MVS_SYNTH_DIR}/train/{SEQ:0000..0119}/
        rgb/%04d.jpg     RGB JPEG, 960x540
        depth/%04d.npy   float32 (540, 960), metric meters; 0 = sky
        cam/%04d.npz     keys "intrinsics" (3,3) float32, "pose" (4,4) float64

Conventions used here (validated empirically against this copy, not assumed):

* Depth is already metric meters (scale 1.0; reprojected depth matches the
  pose-translation scale to ~0.1-0.2% median error). Sky is encoded as **0**
  (the original EXR ``inf`` was converted to 0 in this copy; no nan/inf remain).
  We map 0 / negative / non-finite pixels to the repo-wide sky sentinel
  ``-1.0`` BEFORE ``process_one_image`` (DEPTH convention: 0 = invalid,
  < 0 = sky) and advertise SKY_MASK. Valid depth reaches ~7800 m for distant
  GTA-V terrain.
* ``cam/%04d.npz["pose"]`` is **camera-to-world** with OpenCV camera axes
  (x right, y down, z forward); world->camera is ``np.linalg.inv(pose)[:3, :4]``
  (full inverse, matching the verified survey recipe -- the stored rotations
  carry ~1e-5 non-orthonormality, so the rigid-transpose shortcut is avoided).
  Cross-frame depth reprojection closes to ~0.1% median relative error,
  confirming the convention.
* ``cam/%04d.npz["intrinsics"]`` is a per-frame (3,3) pinhole K in pixels of
  the native 960x540 frame (fx ~= fy ~= 579, cx = 480, cy = 270 with < 0.25 px
  per-frame jitter). It is used per frame as stored; override via
  ``intrinsics=[fx, fy, cx, cy]`` if needed.
* Sequences are temporally ordered video (smooth ~1-5 m/frame motion), but the
  dataset ships no timestamps, so TIMESTAMP is not advertised.

Known quirks (from the on-disk survey):

* World coordinates are raw GTA-V map coordinates (translations ~1e3-1e4), so
  float32 world points carry ~1e-3 m quantization; harmless for relative
  losses/metrics, but consider per-sequence re-centering for absolute uses.
* A few sequences start stationary (e.g. seq 0000 frames 0/1 share a pose), so
  degenerate pairs are possible; inter-frame motion can reach ~10 deg / ~4.6 m.

Construction lists only the sequence NAMES (one directory listing); per-frame
enumeration is deferred to the first access of each sequence and cached.
Because of that deferral, a sequence with fewer than ``min_num_images`` frames
surfaces as a ``ValueError`` at first access instead of being silently skipped
at construction.
"""
from __future__ import annotations

import fnmatch
import glob
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class MvsSynthDataset(BaseDataset):
    """MVS-Synth (GTA-V synthetic) as a VGGT-Omega BaseDataset (video sampling,
    metric depth, per-frame GT cameras, sky encoded in depth)."""

    # MVS-Synth provides RGB + metric depth + per-frame GT cameras, with sky
    # encoded in the depth channel. As with the TUM vendor, WORLD_POINTS /
    # CAM_POINTS are only the depth re-projected through the GT poses (not an
    # independent point-cloud GT), so they are NOT advertised as evaluable GT
    # modalities -- process_one_image still computes them (e.g. for
    # depth-supervised point heads), they just must not be scored as a point
    # cloud. No timestamps ship with the data, so TIMESTAMP is not advertised.
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.POINT_MASK,
            Modality.SKY_MASK,
        }
    )

    @staticmethod
    def read_mvs_synth_depth(path: str) -> np.ndarray:
        """Read an MVS-Synth depth ``.npy`` -> float32 (H, W) meters with sky as -1.0.

        This preprocessed copy stores depth as float32 meters with sky encoded
        as 0 (the original EXR ``inf``). 0, negative and non-finite values are
        all mapped to the repo-wide sky sentinel ``-1.0`` (DEPTH convention:
        0 = invalid, < 0 = sky), so downstream masks are ``depth > 0`` (valid)
        and ``depth < 0`` (sky).
        """
        arr = np.load(path)
        depth = np.asarray(arr, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(
                f"MVS-Synth depth {path!r}: expected a 2-D array, got shape {depth.shape}"
            )
        if depth is arr:
            depth = depth.copy()
        depth[(depth <= 0) | ~np.isfinite(depth)] = -1.0
        return depth

    @staticmethod
    def read_mvs_synth_camera(path: str):
        """Read an MVS-Synth ``cam/%04d.npz`` -> ``(K, c2w)``.

        Returns ``K`` as (3,3) float32 pixels (native 960x540 frame) and
        ``c2w`` as the raw (4,4) float64 camera-to-world pose (OpenCV axes).
        Raises ValueError on missing keys or wrong shapes.
        """
        with np.load(path) as cam:
            try:
                K = np.asarray(cam["intrinsics"], dtype=np.float32)
                c2w = np.asarray(cam["pose"], dtype=np.float64)
            except KeyError as e:
                raise ValueError(
                    f"MVS-Synth camera {path!r}: missing key {e} "
                    f"(have {sorted(cam.keys())})"
                ) from e
        if K.shape != (3, 3):
            raise ValueError(f"MVS-Synth camera {path!r}: intrinsics shape {K.shape} != (3, 3)")
        if c2w.shape != (4, 4):
            raise ValueError(f"MVS-Synth camera {path!r}: pose shape {c2w.shape} != (4, 4)")
        return K, c2w

    @staticmethod
    def mvs_synth_pose_to_w2c(c2w) -> np.ndarray:
        """(4,4) camera-to-world (OpenCV axes) -> world-to-camera (3,4) float32.

        Uses the survey-verified recipe ``np.linalg.inv(pose)[:3, :4]`` -- a
        full matrix inverse rather than the rigid ``[R^T | -R^T t]`` shortcut,
        because the stored rotations carry ~1e-5 non-orthonormality. Raises
        ValueError on a non-(4,4) or non-finite pose.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"MVS-Synth pose: expected (4, 4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("MVS-Synth pose is non-finite")
        return np.linalg.inv(c2w)[:3, :4].astype(np.float32)

    @staticmethod
    def mvs_synth_intrinsics(K, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K in pixels of the native 960x540 frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the per-frame ``K`` from
        the cam file is validated and returned. Raises ValueError when neither
        is usable.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        if K is None:
            raise ValueError("MVS-Synth intrinsics: no K given and no override")
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"MVS-Synth intrinsics: expected (3, 3), got {K.shape}")
        return K

    def _list_frames(self, seq_name: str) -> list:
        """Frame table for ``seq_name`` -> [(rgb_path, depth_path, cam_path,
        frame_idx)], ordered by frame index. Enumerated lazily on first access
        and cached (construction never lists frames). Raises ValueError when
        the sequence holds fewer than ``min_num_images`` frames."""
        cached = self._frames_cache.get(seq_name)
        if cached is not None:
            return cached
        seq_dir = self.data_store[seq_name]
        frames = []
        for rgb_path in glob.glob(os.path.join(seq_dir, "rgb", "*.jpg")):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]  # "0042"
            if not stem.isdigit():
                continue
            frames.append(
                (
                    rgb_path,
                    os.path.join(seq_dir, "depth", stem + ".npy"),
                    os.path.join(seq_dir, "cam", stem + ".npz"),
                    int(stem),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if len(frames) < self.min_num_images:
            # Frame enumeration is deferred, so an undersized sequence cannot be
            # skipped at construction; fail loudly instead of under-filling V.
            raise ValueError(
                f"MVS-Synth sequence {seq_name!r}: only {len(frames)} frames "
                f"(< min_num_images={self.min_num_images})"
            )
        self._frames_cache[seq_name] = frames
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        MVS_SYNTH_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if MVS_SYNTH_DIR is None:
            raise ValueError("MVS_SYNTH_DIR must be specified")
        self.MVS_SYNTH_DIR = MVS_SYNTH_DIR
        self.expand_ratio = expand_ratio
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # The on-disk copy ships a single "train/" split dir; accept either the
        # dataset root or the split dir itself.
        split_dir = os.path.join(MVS_SYNTH_DIR, "train")
        if not os.path.isdir(split_dir):
            split_dir = MVS_SYNTH_DIR
        self.split_dir = split_dir

        # Cheap construction: ONE directory listing for sequence names; frame
        # tables are built lazily per sequence in _list_frames.
        patterns = sequences or ["*"]
        try:
            entries = sorted(e.name for e in os.scandir(split_dir) if e.is_dir())
        except FileNotFoundError:
            entries = []
        self.data_store = {
            name: os.path.join(split_dir, name)
            for name in entries
            if any(fnmatch.fnmatch(name, pat) for pat in patterns)
        }

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable MVS-Synth sequences under {MVS_SYNTH_DIR} "
                f"(sequences={patterns})"
            )
        self._frames_cache = {}
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration). Triggers the
        lazy frame enumeration for that one sequence."""
        return len(self._list_frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self._list_frames(name)[0][0]
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
        frames = self._list_frames(seq_name)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, original_sizes = [], []

        for i in ids:
            rgb_path, depth_path, cam_path, _ = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Globbed from the sequence dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"MVS-Synth: could not read image {rgb_path}"
                )
            depth_map = self.read_mvs_synth_depth(depth_path)  # sky already -1.0
            K_raw, c2w = self.read_mvs_synth_camera(cam_path)
            pose_w2c = self.mvs_synth_pose_to_w2c(c2w)
            K = self.mvs_synth_intrinsics(K_raw, self.intrinsics_override)
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
            sky_masks.append(depth_map < 0)  # sky was mapped to -1.0 before processing
            original_sizes.append(original_size)

        return {
            "seq_name": "mvs_synth_" + seq_name,
            "ids": np.array(ids),
            "frame_num": len(images),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "sky_masks": sky_masks,
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
