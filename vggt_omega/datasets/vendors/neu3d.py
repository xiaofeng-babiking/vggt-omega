"""Neu3D (DyNeRF) vendor for the VGGT-Omega dataset API.

This cluster's Neu3D copy is a single scene with a single camera -- a 300-frame
monocular video of the DyNeRF ``cut_roasted_beef`` scene from ``cam00`` -- with
pre-extracted PNG frames and NOTHING else (no mp4s, no other cam01..cam20, no
``poses_bounds.npy``, no depth)::

    <root>/cut_roasted_beef/cam00/
        images/0000.png .. 0299.png          8-bit RGB PNG, 1352x1014 (W x H)
        downsampled_2x/0000.png .. 0299.png  exact 2x downsample, 676x507
        gt/{FFFF}_png/{prompt}.jpg           open-vocab segmentation GT (see below)

Conventions used here (validated empirically against this dataset, not assumed):

* **IMAGE-ONLY vendor.** There is no depth, no pose and no intrinsics anywhere
  on disk (full-tree extension tally: only png/jpg), so only IMAGE is advertised
  as a modality. ``get_data`` still returns the full core key set so
  ``ComposedDataset._tensorize``'s fixed schema keeps working:

  - all-zero float32 depth maps (0 = invalid everywhere) -> ``point_masks`` all
    False and zeroed world/cam points;
  - **identity** (3,4) world->camera extrinsics (a single static camera; the
    world frame is simply the camera frame);
  - a plausible **placeholder** pinhole K (focal = max(H, W) px, principal
    point = image center) so ``process_one_image`` crop/resize geometry works.

  EXTRINSICS / INTRINSICS are NOT advertised: the values are placeholders, not
  GT, and must never be scored. No ``sky_masks`` key is emitted: SKY_MASK is not
  advertised and an all-False mask for a kitchen scene with a glass window /
  unknown depth would be unverifiable GT (same rule as DL3DV/MegaDepth).
* ``images/`` (1352x1014) is itself a 2x downsample of the original Neu3D
  2704x2028 capture; ``downsampled_2x/`` (676x507) is 2x relative to
  ``images/``. Select via ``image_variant`` ("images" | "downsampled_2x").
* Frames are a temporally ordered video from a static camera filming a dynamic
  scene (mean |f0-f1| = 1.5 vs |f0-f150| = 5.1 gray levels): ``is_video=True``.
  Scene scale is unknown (no metric anchor): ``is_metric=False``. There are no
  per-frame timestamps on disk and this re-export documents no capture rate, so
  TIMESTAMP is not advertised (we refuse to fabricate one).
* **Out of scope:** ``gt/{frame}_png/`` holds open-vocabulary segmentation GT
  for 17 keyframes (0000, 0020, ..., 0260, 0267, 0280, 0299) -- five
  text-prompt-named binary JPEG masks each (threshold > 127; verified cleanly
  bimodal). It could be exposed as SEMANTIC+TEXT on 17/300 frames if the eval
  harness ever supports prompt-conditioned masks, but it is irrelevant to the
  geometry pipeline and deliberately not loaded here.

Construction is trivially fast (one scene, 300 frames): frame paths are listed
eagerly per sequence like TUM/7-Scenes; pixels are only decoded for the frames
actually sampled.
"""
from __future__ import annotations

import fnmatch
import glob
import logging
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class Neu3dDataset(BaseDataset):
    """Neu3D cut_roasted_beef/cam00 as a VGGT-Omega BaseDataset (image-only video)."""

    # Neu3D (this copy) provides RGB frames and NOTHING else: no depth, no poses,
    # no intrinsics. DEPTH/POINT_MASK/SKY_MASK cannot be supervised or scored;
    # EXTRINSICS/INTRINSICS are emitted only as identity/center-pp placeholders
    # for ComposedDataset's fixed tensorization schema and must not be advertised
    # (they are not GT); WORLD_POINTS/CAM_POINTS are never advertised.
    AVAILABLE = frozenset({Modality.IMAGE})

    _IMAGE_VARIANTS = ("images", "downsampled_2x")

    @staticmethod
    def identity_extrinsics() -> np.ndarray:
        """Identity (3,4) float32 world->camera extrinsics -- Neu3D ships no
        poses and has a single static camera, so the world frame is defined as
        the camera frame (placeholder, NOT GT; EXTRINSICS is not advertised)."""
        return np.concatenate(
            [np.eye(3), np.zeros((3, 1))], axis=1
        ).astype(np.float32)

    @staticmethod
    def placeholder_intrinsics(height: int, width: int, override=None) -> np.ndarray:
        """(3,3) float32 placeholder pinhole K for an image of (height, width).

        Neu3D ships no calibration, so a plausible default is synthesized:
        focal = max(H, W) px (a generic ~53 deg horizontal FoV), principal
        point = image center -- enough for process_one_image crop/resize
        geometry (placeholder, NOT GT; INTRINSICS is not advertised).
        ``override``=[fx, fy, cx, cy] wins. Raises ValueError on a non-positive
        image size.
        """
        if override is not None:
            fx, fy, cx, cy = override
        else:
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"Neu3D intrinsics: image size must be positive, got ({height}, {width})"
                )
            fx = fy = float(max(height, width))
            cx, cy = width / 2.0, height / 2.0
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )

    @staticmethod
    def empty_depth(height: int, width: int) -> np.ndarray:
        """All-zero float32 (H,W) depth map -- Neu3D ships no depth, and in the
        DEPTH convention 0 means invalid, so every pixel is 'no measurement'
        (point_masks all False)."""
        return np.zeros((int(height), int(width)), dtype=np.float32)

    @staticmethod
    def _list_frames(frames_dir: str) -> list:
        """List a ``<cam_dir>/<image_variant>`` dir -> [(rgb_path, frame_num)],
        ordered by frame number (filenames are zero-padded ints, ``0000.png``).
        Non-numeric stems are ignored; a missing dir lists as empty, which makes
        the sequence fall below ``min_num_images`` and get skipped at
        construction (the TUM/7-Scenes warn-and-skip contract)."""
        frames = []
        for rgb_path in glob.glob(os.path.join(frames_dir, "*.png")):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]  # "0000"
            if not stem.isdigit():
                continue
            frames.append((rgb_path, int(stem)))
        frames.sort(key=lambda fr: fr[1])
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        NEU3D_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        image_variant: str = "images",
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if NEU3D_DIR is None:
            raise ValueError("NEU3D_DIR must be specified")
        if image_variant not in self._IMAGE_VARIANTS:
            raise ValueError(
                f"image_variant must be one of {self._IMAGE_VARIANTS}, got {image_variant!r}"
            )
        self.NEU3D_DIR = NEU3D_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.image_variant = image_variant
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # Enumerate "<scene>/<cam>" sequences (this copy has exactly one:
        # cut_roasted_beef/cam00). `sequences` patterns match either the full
        # "<scene>/<cam>" name or the scene name alone.
        cam_dirs = sorted(
            d
            for d in glob.glob(os.path.join(NEU3D_DIR, "*", "cam*"))
            if os.path.isdir(d)
        )
        all_names = [
            os.path.join(
                os.path.basename(os.path.dirname(d)), os.path.basename(d)
            ).replace(os.sep, "/")
            for d in cam_dirs
        ]
        patterns = sequences or ["*"]
        names = sorted(
            {
                n
                for n in all_names
                for pat in patterns
                if fnmatch.fnmatch(n, pat) or fnmatch.fnmatch(n.split("/", 1)[0], pat)
            }
        )

        self.data_store = {}
        for name in names:
            frames = self._list_frames(
                os.path.join(NEU3D_DIR, name, image_variant)
            )
            if len(frames) < min_num_images:
                logging.warning(
                    "Neu3D seq %s: only %d frames (< %d); skipping",
                    name, len(frames), min_num_images,
                )
                continue
            self.data_store[name] = frames

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable Neu3D sequences under {NEU3D_DIR} (sequences={patterns})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached
        (depends on ``image_variant``: 1014x1352 full / 507x676 downsampled)."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self.data_store[name][0][0]
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
        frames = self.data_store[seq_name]

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            rgb_path, _frame_num = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Globbed from the sequence dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(f"Neu3D: could not read image {rgb_path}")
            original_size = np.array(image.shape[:2])
            # No camera/depth exist for Neu3D: identity pose, placeholder K
            # (focal = max(H,W), pp = center), all-zero depth (0 = invalid).
            pose_w2c = self.identity_extrinsics()
            K = self.placeholder_intrinsics(*original_size, override=self.intrinsics_override)
            depth_map = self.empty_depth(*original_size)

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
            # With zero depth, pmask is all False but the world-point unprojection
            # broadcasts the camera center everywhere; zero the masked-out points
            # so undefined geometry reads as zeros, not fake points.
            world_p[~pmask] = 0.0
            cam_p[~pmask] = 0.0

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri)
            intrinsics.append(intri)
            cam_points.append(cam_p)
            world_points.append(world_p)
            point_masks.append(pmask)
            original_sizes.append(original_size)

        return {
            "seq_name": "neu3d_" + seq_name,
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
            # would be unverifiable GT (no depth exists to separate sky).
            "original_sizes": original_sizes,
            "is_metric": False,  # no metric anchor anywhere on disk
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
