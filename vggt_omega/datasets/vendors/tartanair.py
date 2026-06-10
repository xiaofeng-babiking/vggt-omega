"""TartanAir vendor for the VGGT-Omega dataset API.

TartanAir is a synthetic (AirSim) photorealistic SLAM dataset; depth and poses
are exact simulator ground truth. The dump under ``TARTANAIR_DIR`` is a
PREPROCESSED, FLATTENED copy of the official release -- there are no
``image_left``/``depth_left`` directories and no ``pose_left.txt``. Instead::

    <TARTANAIR_DIR>/train/<env>/<Easy|Hard>/P0XX/
        NNNNNN_rgb.png      RGB uint8, 640x480
        NNNNNN_depth.npy    float32 (480, 640), metric METERS (scale 1.0)
        NNNNNN_cam.npz      'camera_pose'       (4,4) camera-to-world
                            'camera_intrinsics' (3,3) pinhole K

A sequence is ``<env>/<Easy|Hard>/P0XX`` (369 total across 18 environments;
``P`` indices are non-contiguous per env, so sequences are discovered by glob,
never by ``range()``). Frame indices within a sequence are contiguous 6-digit
counters from ``000000``.

Conventions used here (validated empirically against this dump, not assumed):

* ``camera_pose`` is camera-to-world ALREADY in the OpenCV optical frame
  (x-right, y-down, z-forward): the official NED -> OpenCV axis remap was
  applied during preprocessing. Do NOT apply the textbook TartanAir NED remap
  again -- it corrupts the poses. World->camera is the plain rigid inverse
  ``[R^T | -R^T t]``; cross-frame depth reprojection closes to ~0.03-0.14%
  median relative error across environments, confirming the convention.
* Depth is exact simulator GT in meters (no NaN/Inf/zero/negative values
  observed). Sky in outdoor scenes is encoded as VERY LARGE finite depth
  (> 10000 m, max ~16300 m; real geometry stays below ~1000 m and the band
  (1000, 10000] is essentially empty). ``decode_tartanair_depth`` maps
  depth > ``sky_threshold`` to -1.0 (the repo-wide sky code), and the rare
  ambiguous band (``valid_max``, ``sky_threshold``] plus any non-finite /
  negative values to 0.0 (invalid). SKY_MASK is therefore advertised.
* Intrinsics are globally constant: fx = fy = 320, cx = 320, cy = 240 for the
  native 640x480 frame (single unique K across every sampled sequence; the
  per-frame ``camera_intrinsics`` in each ``cam.npz`` merely duplicates it).
  Override via ``intrinsics=[fx, fy, cx, cy]`` if needed.
* This dump ships NO timestamps (the official extras were stripped); frames
  are temporally ordered by the 6-digit index at a fixed simulator rate, so
  ``is_video=True`` but TIMESTAMP is NOT advertised (we refuse to fabricate a
  clock with no documented frame rate).

Frame lists are enumerated LAZILY per sequence: with 369 sequences of 300-2300
frames each on a network filesystem, eagerly listing every sequence directory
at construction would dominate startup. Construction only discovers sequence
NAMES (one shallow glob over ``train/*/*/P*``); the per-frame triplet listing
happens on first access to a sequence and is cached. Consequently the
``min_num_images`` check is also lazy: a too-thin sequence raises a clear
ValueError when first touched instead of being silently dropped up front.
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


class TartanAirDataset(BaseDataset):
    """TartanAir as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # Globally constant TartanAir pinhole intrinsics (fx, fy, cx, cy) for the
    # native 640x480 frame (verified identical across all sampled sequences).
    _FX = 320.0
    _FY = 320.0
    _PRINCIPAL_POINT = (320.0, 240.0)

    # TartanAir provides RGB + simulator-exact metric depth + GT camera poses.
    # As with the TUM vendor, WORLD_POINTS / CAM_POINTS are only the depth
    # re-projected through the GT poses (not an independent point-cloud GT), so
    # they are NOT advertised as evaluable GT modalities -- process_one_image
    # still computes them (e.g. for depth-supervised point heads), they just
    # must not be scored as a point cloud. No timestamps exist in this dump.
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
    def tartanair_pose_to_w2c(c2w) -> np.ndarray:
        """TartanAir ``camera_pose`` (4,4) camera-to-world -> world-to-camera (3,4) OpenCV.

        The dump's poses are ALREADY in OpenCV camera axes (the NED remap was
        applied during preprocessing, verified by cross-frame depth
        reprojection), so this is the plain rigid inverse ``[R^T | -R^T t]`` --
        no axis flip. Raises ValueError on a wrong shape or non-finite pose.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"TartanAir camera_pose: expected (4,4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("TartanAir camera_pose is non-finite")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def decode_tartanair_depth(
        arr, valid_max: float = 1000.0, sky_threshold: float = 10000.0
    ) -> np.ndarray:
        """Raw TartanAir depth array -> float32 (H,W) meters in repo convention.

        TartanAir encodes sky as very large finite depth (> ~10000 m) rather
        than 0/NaN/Inf. Mapping: depth > ``sky_threshold`` (or +Inf) -> -1.0
        (sky); non-finite, negative, or in the ambiguous (``valid_max``,
        ``sky_threshold``] band -> 0.0 (invalid); everything else stays as-is
        (already metric meters, scale 1.0).
        """
        depth = np.array(arr, dtype=np.float32, copy=True)
        sky = depth > sky_threshold  # +Inf included, NaN excluded (NaN > x is False)
        invalid = ~np.isfinite(depth) | (depth < 0) | ((depth > valid_max) & ~sky)
        depth[invalid] = 0.0
        depth[sky] = -1.0
        return depth

    @classmethod
    def read_tartanair_depth(
        cls, path: str, valid_max: float = 1000.0, sky_threshold: float = 10000.0
    ) -> np.ndarray:
        """Read a TartanAir ``*_depth.npy`` -> float32 (H,W) meters, sky -> -1.0."""
        return cls.decode_tartanair_depth(np.load(path), valid_max, sky_threshold)

    @classmethod
    def read_tartanair_cam(cls, path: str) -> np.ndarray:
        """Read a TartanAir ``*_cam.npz`` -> world-to-camera (3,4) float32 OpenCV.

        Raises ValueError if the archive lacks the ``camera_pose`` key or the
        pose is malformed/non-finite.
        """
        with np.load(path) as cam:
            if "camera_pose" not in cam:
                raise ValueError(
                    f"TartanAir cam file {path!r}: missing 'camera_pose' key "
                    f"(has {sorted(cam.keys())})"
                )
            return cls.tartanair_pose_to_w2c(cam["camera_pose"])

    @classmethod
    def tartanair_intrinsics(cls, override=None) -> np.ndarray:
        """(3,3) pinhole K for TartanAir's native 640x480 frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the globally constant
        fx=fy=320, principal point=(320,240) is used. Raises ValueError if the
        override does not hold exactly four values.
        """
        if override is not None:
            if len(override) != 4:
                raise ValueError(
                    f"TartanAir intrinsics override must be [fx, fy, cx, cy], got {override!r}"
                )
            fx, fy, cx, cy = override
        else:
            fx, fy = cls._FX, cls._FY
            cx, cy = cls._PRINCIPAL_POINT
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def _match_sequences(names: list[str], patterns) -> list[str]:
        """Filter ``env/Difficulty/P0XX`` names by glob ``patterns``.

        A pattern matches a sequence if it matches the full name or any
        directory prefix of it, so ``"abandonedfactory"`` selects every
        sequence of that env, ``"*/Easy"`` every Easy sequence, and
        ``"abandonedfactory/Easy/P000"`` exactly one.
        """
        if not patterns:
            return list(names)
        out = []
        for name in names:
            candidates = (name, *(name.rsplit("/", k)[0] for k in (1, 2)))
            if any(fnmatch.fnmatch(c, pat) for pat in patterns for c in candidates):
                out.append(name)
        return out

    def _get_frames(self, seq_name: str) -> list:
        """Frame triplets for ``seq_name`` -> [(rgb_path, depth_path, cam_path,
        frame_idx)], ordered by frame index. Listed lazily on first access and
        cached (369 sequences x 300-2300 frames would dominate construction).

        Raises ValueError if the sequence holds fewer than ``min_num_images``
        frames (this dump's thinnest sequence has ~300, so this firing
        indicates a corrupt/partial copy -- fail loudly rather than emit
        duplicate-heavy batches).
        """
        if seq_name in self._frames_cache:
            return self._frames_cache[seq_name]
        seq_dir = self.data_store[seq_name]
        frames = []
        for rgb_path in glob.glob(os.path.join(seq_dir, "*_rgb.png")):
            stem = os.path.basename(rgb_path)[: -len("_rgb.png")]  # "000123"
            frames.append(
                (
                    rgb_path,
                    os.path.join(seq_dir, stem + "_depth.npy"),
                    os.path.join(seq_dir, stem + "_cam.npz"),
                    int(stem),
                )
            )
        frames.sort(key=lambda fr: fr[3])
        if len(frames) < self.min_num_images:
            raise ValueError(
                f"TartanAir sequence {seq_name!r}: only {len(frames)} frames "
                f"(< min_num_images={self.min_num_images}); corrupt or partial copy?"
            )
        self._frames_cache[seq_name] = frames
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        TARTANAIR_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        valid_max: float = 1000.0,
        sky_threshold: float = 10000.0,
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if TARTANAIR_DIR is None:
            raise ValueError("TARTANAIR_DIR must be specified")
        self.TARTANAIR_DIR = TARTANAIR_DIR
        self.expand_ratio = expand_ratio
        self.valid_max = valid_max
        self.sky_threshold = sky_threshold
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # The dump has a single top-level train/ dir; also accept a root that
        # already points inside it.
        root = os.path.join(TARTANAIR_DIR, "train")
        if not os.path.isdir(root):
            root = TARTANAIR_DIR

        # Discover sequence NAMES only (one shallow glob; ~50 dir listings for
        # the full 369-sequence dump). Per-frame listing is deferred to
        # _get_frames so construction stays fast on a network FS.
        all_names = sorted(
            os.path.relpath(d, root)
            for d in glob.glob(os.path.join(root, "*", "*", "P*"))
            if os.path.isdir(d)
        )
        names = self._match_sequences(all_names, sequences)

        # Lightweight handles: name -> sequence dir; frames are cached lazily.
        self.data_store = {name: os.path.join(root, name) for name in names}
        self._frames_cache = {}

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable TartanAir sequences under {TARTANAIR_DIR} "
                f"(sequences={sequences})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration). Triggers the
        lazy per-sequence frame listing for that one sequence."""
        return len(self._get_frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self._get_frames(name)[0][0]
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
        frames = self._get_frames(seq_name)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)
        K = self.tartanair_intrinsics(self.intrinsics_override)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, original_sizes = [], []

        for i in ids:
            rgb_path, depth_path, cam_path, _frame_idx = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Globbed from the sequence dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"TartanAir: could not read image {rgb_path}"
                )
            depth_map = self.read_tartanair_depth(
                depth_path, self.valid_max, self.sky_threshold
            )
            pose_w2c = self.read_tartanair_cam(cam_path)
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
            "seq_name": "tartanair_" + seq_name,
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
