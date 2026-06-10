"""NYU Depth V2 (Eigen test split) vendor for the VGGT-Omega dataset API.

This loads the 654-frame NYU Depth V2 test set (the standard Eigen split used
for monocular-depth evaluation) from a pre-decoded copy laid out as::

    {NYU_DIR}/val/nyu_images/XXXXX.png   RGB, 640x480 (654 frames)
    {NYU_DIR}/val/nyu_depths/XXXXX.npy   float32 (480,640) depth, paired by basename

Basenames are the non-contiguous original h5 ids (00001, 00002, 00009, ...), so
frames are enumerated by sorted glob, never by ``range()``. The sibling dir
families on disk are redundant re-decodes and are deliberately ignored:
``official_data/`` and ``office_10/`` hold the same 654 frames renamed
000000..000653 in an UNSORTED glob order (their indices do NOT correspond to
the ``nyu_*`` basenames), ``nyu_depth_imgs/`` is per-image min-max-normalized
uint8 visualization (useless for metric eval), and ``official/*.h5`` needs
h5py while being fully redundant with the decoded copies.

Conventions used here (validated empirically against this dataset, not assumed):

* Frames are 654 INDEPENDENT single RGB-D captures with no temporal structure;
  each frame is modeled as its own 1-frame sequence (``min_num_images``
  defaults to 1, ``is_video=False``). Drawing V > 1 frames with
  ``allow_duplicate_img`` simply duplicates the single frame.
* Depth: ``np.load`` gives float32 meters directly (scale 1.0), dense
  Kinect-projected depth with no invalid pixels (verified global range
  0.713-9.987 m over all 654 files) -- the labeled/filled split.
* No poses exist on disk (the frames are unrelated captures); extrinsics are
  emitted as the identity (3,4) world->camera so each frame's world frame is
  its own camera frame. EXTRINSICS is NOT advertised as an evaluable modality.
* No intrinsics exist on disk either; the standard published NYUv2 Kinect
  calibration (fx=518.857901, fy=519.469611, cx=325.582245, cy=253.736166 for
  the native 640x480 frame) is hardcoded so depth can be unprojected to
  points. INTRINSICS is NOT advertised (literature calibration, not on-disk
  GT). Override via ``intrinsics=[fx, fy, cx, cy]`` if needed.
* Indoor and sky-free, so SKY_MASK is advertised with all-False masks
  (depth < 0 convention), matching the TUM/7-Scenes indoor convention.
* No timestamps or camera ids on disk; neither is advertised nor fabricated.

Official eval-crop convention (documented, NOT applied here): the standard
Eigen evaluation crops to rows 45:471, cols 41:601 of the 480x640 frame before
scoring (the border region is unreliable Kinect projection / white-border
fill). The crop is not stored on disk; it is exposed as
``NyuDataset.EIGEN_CROP`` / ``NyuDataset.eigen_crop_mask()`` for eval harnesses
that follow the standard protocol, but this vendor emits the full frame.
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


class NyuDataset(BaseDataset):
    """NYU Depth V2 test split as a VGGT-Omega BaseDataset (independent 1-frame
    sequences, metric depth)."""

    # Standard published NYUv2 Kinect color-camera pinhole intrinsics
    # (fx, fy, cx, cy) for the native 640x480 frame. NOT stored on disk --
    # literature calibration, so INTRINSICS is not advertised as GT.
    _FX = 518.857901
    _FY = 519.469611
    _CX = 325.582245
    _CY = 253.736166

    # Standard Eigen evaluation crop of the 480x640 frame: rows [45, 471),
    # cols [41, 601). Documented convention only -- NOT applied by this vendor.
    EIGEN_CROP = (45, 471, 41, 601)

    # NYU provides RGB + dense metric depth only. Poses do not exist (identity
    # extrinsics are emitted but are not GT) and intrinsics are hardcoded from
    # the literature (not on-disk GT), so neither EXTRINSICS nor INTRINSICS is
    # advertised. WORLD_POINTS / CAM_POINTS are never advertised (they are only
    # the depth unprojected through the hardcoded K). SKY_MASK is advertised
    # all-False per the indoor sky-free convention (TUM/7-Scenes). No
    # timestamps, no camera ids.
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.POINT_MASK,
            Modality.SKY_MASK,
        }
    )

    @staticmethod
    def identity_extrinsic() -> np.ndarray:
        """Identity world->camera (3,4) float32. NYU frames are independent
        captures with no pose GT, so each frame's world frame is defined to be
        its own camera frame."""
        return np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float32)

    @staticmethod
    def read_nyu_depth(path: str) -> np.ndarray:
        """Read a NYU ``.npy`` depth map -> float32 (H,W) meters (stored scale 1.0).

        Non-finite or negative values map to 0 (invalid) defensively; the NYU
        val depth is dense filled Kinect depth and contains neither in practice
        (verified 0.713-9.987 m, zero non-positive pixels across all 654 files).
        """
        depth = np.load(path).astype(np.float32)
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        return depth

    @classmethod
    def nyu_intrinsics(cls, override=None) -> np.ndarray:
        """(3,3) pinhole K for NYU's native 640x480 color frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the standard published
        NYUv2 Kinect calibration is used (literature values -- nothing is
        stored on disk).
        """
        if override is not None:
            fx, fy, cx, cy = override
        else:
            fx, fy, cx, cy = cls._FX, cls._FY, cls._CX, cls._CY
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    @classmethod
    def eigen_crop_mask(cls, shape=(480, 640)) -> np.ndarray:
        """(H,W) bool mask of the standard Eigen evaluation crop (True inside
        rows [45, 471), cols [41, 601) at native 480x640). Convenience for eval
        harnesses; this vendor does NOT apply it."""
        top, bottom, left, right = cls.EIGEN_CROP
        mask = np.zeros(shape, dtype=bool)
        mask[top:bottom, left:right] = True
        return mask

    @staticmethod
    def list_nyu_frames(images_dir: str, depths_dir: str) -> list:
        """Pair RGB PNGs with depth NPYs by basename.

        Returns ``[(name, rgb_path, depth_path)]`` sorted by name (the original
        non-contiguous h5 ids). Frames whose depth file is missing are skipped
        with a warning. Depths are globbed once into a set (not stat'ed per
        frame) to keep construction fast on a network FS.
        """
        have_depth = {
            os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(depths_dir, "*.npy"))
        }
        frames = []
        for rgb_path in sorted(glob.glob(os.path.join(images_dir, "*.png"))):
            name = os.path.splitext(os.path.basename(rgb_path))[0]
            depth_path = os.path.join(depths_dir, name + ".npy")
            if name not in have_depth:
                logging.warning("NYU frame %s: missing depth %s; skipping", name, depth_path)
                continue
            frames.append((name, rgb_path, depth_path))
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "test",
        NYU_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        intrinsics=None,
        min_num_images: int = 1,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if NYU_DIR is None:
            raise ValueError("NYU_DIR must be specified")
        self.NYU_DIR = NYU_DIR
        self.expand_ratio = expand_ratio
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        # Only the 654-frame val/ split exists on disk (the official test set);
        # `split` selects the virtual epoch length, not different data.
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        images_dir = os.path.join(NYU_DIR, "val", "nyu_images")
        depths_dir = os.path.join(NYU_DIR, "val", "nyu_depths")
        frames = self.list_nyu_frames(images_dir, depths_dir)
        if sequences:
            frames = [
                fr for fr in frames
                if any(fnmatch.fnmatch(fr[0], pat) for pat in sequences)
            ]

        # Every NYU "sequence" is exactly one independent frame, so any
        # min_num_images > 1 would drop the whole dataset.
        self.data_store = {}
        if min_num_images > 1:
            logging.warning(
                "NYU frames are independent 1-frame sequences; min_num_images=%d "
                "(> 1) drops everything", min_num_images,
            )
        else:
            for name, rgb_path, depth_path in frames:
                self.data_store[name] = [(rgb_path, depth_path)]

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable NYU frames under {NYU_DIR} (sequences={sequences})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (always 1: NYU frames are independent captures)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frame for the sequence at
        ``local_idx``, read lazily from the image header and cached."""
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
            # Each sequence holds exactly 1 frame: a V-frame draw needs
            # allow_duplicate_img to duplicate it (np.random.choice raises
            # otherwise, which is the correct loud failure).
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)
        K = self.nyu_intrinsics(self.intrinsics_override)
        pose_w2c = self.identity_extrinsic()

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, original_sizes = [], []

        for i in ids:
            rgb_path, depth_path = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Globbed from nyu_images/, so the file should exist; fail loudly
                # (a silent skip would yield fewer than img_per_seq frames and
                # break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"NYU: could not read image {rgb_path}"
                )
            depth_map = self.read_nyu_depth(depth_path)
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
            sky_masks.append(depth_map < 0)  # NYU is indoor: always all-False (sky convention = depth<0)
            original_sizes.append(original_size)

        return {
            "seq_name": "nyu_" + seq_name,
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
            "is_video": False,
            "modalities": set(self.available_modalities),
        }
