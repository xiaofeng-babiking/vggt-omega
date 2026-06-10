"""Spring (CVPR'23 synthetic movie benchmark) vendor for the VGGT-Omega dataset API.

This loader targets the PREPROCESSED Spring export on disk, NOT the raw Spring
benchmark layout: there are no ``.dsp5``/h5 disparity files and no stereo right
view in this copy, so everything decodes with numpy + PIL/OpenCV only (h5py is
not needed). Layout::

    {SPRING_DIR}/{split}/{SEQ}/
        rgb/%04d.png     8-bit RGB, 960x540 (half of Spring's native 1920x1080)
        depth/%04d.npy   float32 (540,960) depth in METERS; 0 = invalid
        cam/%04d.npz     'intrinsics': (3,3) float32 K px (for the 960x540 frame)
                         'pose':       (4,4) float32 CAMERA-TO-WORLD, OpenCV axes

Only ``train/`` exists in this copy (37 sequences named ``0001..0047`` WITH
GAPS -- enumerate by directory listing, never by range). Frame indices are
contiguous from 0000 and the three subdirs always hold matching counts
(13..308 frames per sequence, ~5000 frames total).

Conventions used here (validated empirically against this copy, not assumed):

* ``pose`` is camera-to-world in the OpenCV optical frame (x-right, y-down,
  z-forward); world->camera is ``np.linalg.inv(pose)[:3, :4]``. Cross-frame
  depth reprojection closes to ~0.04% median relative error, simultaneously
  confirming the pose convention, the depth-in-meters scale and the intrinsics.
* ``intrinsics`` is stored per frame and is constant within a sequence, but
  varies strongly ACROSS sequences (fx = fy from ~646 to ~2020 px at 960x540;
  cx, cy always (479.75, 269.75) = the image center). It is therefore read from
  the npz per frame -- never assume a global K. Override via
  ``intrinsics=[fx, fy, cx, cy]`` only if you have a recalibration.
* Depth is float32 meters with scale 1.0 (float16-quantized on disk, ~0.1%
  precision); 0 = invalid (fraction varies hugely per frame, up to >60%). Sky
  is a LARGE FINITE scene-dependent depth (e.g. ~780 m), not a sentinel, so no
  reliable SKY_MASK can be derived: it is neither advertised nor is a
  ``sky_masks`` key emitted -- these rendered outdoor scenes DO contain sky,
  so an all-False mask would be wrong GT (same rule as MegaDepth).
* Spring is a rendered movie: smooth constant-rate camera motion (genuine
  video), metric and synthetic-clean, but it ships no capture timestamps in
  this export, so TIMESTAMP is not advertised.

Sequence NAMES are enumerated with a single directory listing at construction;
per-sequence frame lists and all per-frame metadata (cam npz, image headers)
are loaded LAZILY on first access and cached, so construction stays cheap on a
network filesystem. A consequence: ``min_num_images`` cannot drop short
sequences at construction (frame counts are unknown then) -- sequences shorter
than ``min_num_images`` are kept and only warned about on first access (Spring
has 13- and 19-frame sequences below the default of 24; they remain usable
with ``allow_duplicate_img`` or explicit ids). If random sampling without
replacement then asks for more frames than such a sequence has,
``get_data`` raises a ValueError naming the sequence.
"""
from __future__ import annotations

import fnmatch
import logging
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class SpringDataset(BaseDataset):
    """Spring (preprocessed export) as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # Spring provides RGB + metric depth + GT camera poses/intrinsics. As with
    # the TUM vendor, WORLD_POINTS / CAM_POINTS are only the depth re-projected
    # through the GT poses (not an independent point-cloud GT), so they are NOT
    # advertised as evaluable GT modalities -- process_one_image still computes
    # them (e.g. for depth-supervised point heads), they just must not be scored
    # as a point cloud. No SKY_MASK (sky is large finite depth, not a sentinel;
    # no "sky_masks" key is emitted either -- an all-False mask for these
    # outdoor renders would be wrong GT) and no TIMESTAMP (none shipped in
    # this export).
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.POINT_MASK,
        }
    )

    @staticmethod
    def spring_pose_to_w2c(c2w) -> np.ndarray:
        """Spring ``pose`` (4,4) camera-to-world, OpenCV axes -> world-to-camera (3,4) float32.

        Follows the empirically verified recipe ``np.linalg.inv(pose)[:3, :4]``
        (a true inverse, robust to the float32-stored rotation being slightly
        non-orthonormal). Raises ValueError on a wrong shape or non-finite pose.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(
                f"Spring pose: expected a 4x4 camera-to-world matrix, got shape {c2w.shape}"
            )
        if not np.isfinite(c2w).all():
            raise ValueError("Spring pose is non-finite")
        return np.linalg.inv(c2w)[:3, :4].astype(np.float32)

    @staticmethod
    def read_spring_depth(path: str) -> np.ndarray:
        """Read a Spring ``depth/%04d.npy`` -> float32 (H,W) meters; 0 stays 0 (invalid).

        Depth is stored as plain float32 meters (scale 1.0; float16-quantized).
        The on-disk data has no nan/inf/negatives, but any such value is mapped
        to 0 (invalid) defensively to uphold the DEPTH convention.
        """
        arr = np.asarray(np.load(path), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"Spring depth {path!r}: expected a 2-D depth map, got shape {arr.shape}"
            )
        arr[~np.isfinite(arr) | (arr < 0)] = 0.0
        return arr

    @staticmethod
    def spring_intrinsics(K, override=None) -> np.ndarray:
        """Validate/assemble a (3,3) float32 pinhole K for the 960x540 Spring frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the per-frame ``K`` from the
        cam npz is validated (shape (3,3), finite, positive focals) and returned
        as float32. Raises ValueError on malformed intrinsics.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        K = np.asarray(K, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"Spring intrinsics: expected a 3x3 K, got shape {K.shape}")
        if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("Spring intrinsics: K must be finite with positive focals")
        return K

    @classmethod
    def read_spring_cam(cls, path: str, override=None):
        """Read a Spring ``cam/%04d.npz`` -> (K (3,3) float32, w2c (3,4) float32).

        The npz must hold 'intrinsics' and 'pose' (camera-to-world, OpenCV).
        Raises ValueError if either key is missing or malformed.
        """
        with np.load(path) as cam:
            if "intrinsics" not in cam or "pose" not in cam:
                raise ValueError(
                    f"Spring cam {path!r}: expected keys 'intrinsics' and 'pose', "
                    f"got {sorted(cam.files)}"
                )
            K = cls.spring_intrinsics(cam["intrinsics"], override)
            w2c = cls.spring_pose_to_w2c(cam["pose"])
        return K, w2c

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SPRING_DIR: str = None,
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

        if SPRING_DIR is None:
            raise ValueError("SPRING_DIR must be specified")
        self.SPRING_DIR = SPRING_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        split_dir = os.path.join(SPRING_DIR, split)
        if not os.path.isdir(split_dir):
            raise ValueError(
                f"Spring split dir {split_dir} does not exist "
                "(this copy ships only the 'train' split)"
            )
        self._split_dir = split_dir

        # One directory listing for sequence NAMES; frame lists are lazy.
        patterns = sequences or ["*"]
        names = sorted(
            entry.name
            for entry in os.scandir(split_dir)
            if entry.is_dir()
            and any(fnmatch.fnmatch(entry.name, pat) for pat in patterns)
        )
        # name -> frame list, populated lazily on first access (None until then)
        self.data_store = {name: None for name in names}
        self.sequence_list = names
        self.sequence_list_len = len(names)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable Spring sequences under {split_dir} (sequences={patterns})"
            )
        self._native_size_cache = {}

    def _frames(self, seq_name: str) -> list:
        """Frame list ``[(rgb_path, depth_path, cam_path, frame_idx)]`` for one
        sequence, ordered by frame index. Enumerated LAZILY on first access (one
        listing of the sequence's ``rgb/`` dir) and cached in ``data_store``;
        cam npz files are NOT read here.
        """
        frames = self.data_store[seq_name]
        if frames is None:
            seq_dir = os.path.join(self._split_dir, seq_name)
            rgb_dir = os.path.join(seq_dir, "rgb")
            frames = []
            for fname in os.listdir(rgb_dir):
                stem, ext = os.path.splitext(fname)
                if ext.lower() != ".png" or not stem.isdigit():
                    continue
                frames.append(
                    (
                        os.path.join(rgb_dir, fname),
                        os.path.join(seq_dir, "depth", stem + ".npy"),
                        os.path.join(seq_dir, "cam", stem + ".npz"),
                        int(stem),
                    )
                )
            frames.sort(key=lambda fr: fr[3])
            if not frames:
                raise ValueError(f"Spring seq {seq_name}: no rgb frames under {rgb_dir}")
            if len(frames) < self.min_num_images:
                # Frame lists are enumerated lazily, so short sequences cannot be
                # dropped at construction like in the eager vendors; keep + warn.
                logging.warning(
                    "Spring seq %s: only %d frames (< %d); kept (lazy enumeration) -- "
                    "sample with allow_duplicate_img or explicit ids",
                    seq_name, len(frames), self.min_num_images,
                )
            self.data_store[seq_name] = frames
        return frames

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration); lazy + cached."""
        return len(self._frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self._frames(name)[0][0]
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
        frames = self._frames(seq_name)

        if ids is None:
            if not self.allow_duplicate_img and img_per_seq > len(frames):
                # The eager vendors (tum.py) drop sub-min_num_images sequences at
                # construction; lazy enumeration cannot, so name the offender here
                # instead of letting np.random.choice raise its opaque
                # "Cannot take a larger sample than population" error.
                raise ValueError(
                    f"Spring seq {seq_name}: cannot sample {img_per_seq} distinct "
                    f"frames from only {len(frames)} available "
                    "(allow_duplicate_img=False). Use allow_duplicate_img=True, "
                    "explicit ids, or exclude this short sequence via `sequences`."
                )
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            rgb_path, depth_path, cam_path, _frame_idx = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Enumerated from the rgb dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(f"Spring: could not read image {rgb_path}")
            depth_map = self.read_spring_depth(depth_path)
            # Intrinsics vary strongly across sequences: read K per frame from
            # the cam npz (constant within a sequence, but never assume globally).
            K, pose_w2c = self.read_spring_cam(cam_path, self.intrinsics_override)
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
            original_sizes.append(original_size)

        return {
            "seq_name": "spring_" + seq_name,
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
            # would be wrong GT for these outdoor renders (sky is finite depth).
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
