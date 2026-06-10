"""CO3Dv2 (DUSt3R-style preprocessed copy) vendor for the VGGT-Omega dataset API.

This copy of CO3Dv2 is the *preprocessed* (DUSt3R/CroCo-style ``processed_co3d``)
release, not the raw official CO3Dv2: poses are already converted from PyTorch3D
to OpenCV camera-to-world, images are resized to short-side ~384, and only the
selected frames are kept. Layout::

    {CO3D_DIR}/{category}/{seq_id}/
        images/frame{i:06d}.jpg              RGB JPEG (short side ~384; portrait or
                                             landscape per sequence; H,W vary
                                             SLIGHTLY frame-to-frame)
        images/frame{i:06d}.npz              per-frame metadata:
                                             - camera_pose       (4,4) float32 cam2world
                                             - camera_intrinsics (3,3) float64 pinhole
                                             - maximum_depth     () float32 depth scale
        depths/frame{i:06d}.jpg.geometric.png  uint16 PNG, same HxW as the RGB
        masks/frame{i:06d}.png               soft foreground probability (unused here)
    {CO3D_DIR}/selected_seqs_train.json      dict category -> seq_id -> [frame idx]
    {CO3D_DIR}/selected_seqs_test.json       same structure (51 categories, 2511 seqs)

Conventions used here (validated empirically against this copy, not assumed):

* ``camera_pose`` is **camera-to-world in the OpenCV optical frame** (x-right,
  y-down, z-forward); world->camera is ``np.linalg.inv(camera_pose)[:3, :4]``.
  Cross-frame depth reprojection closes to ~0.1% median relative error,
  confirming the convention.
* Depth PNGs are **per-frame normalized** uint16: the integer value alone is
  meaningless and MUST be rescaled by that frame's ``maximum_depth`` scalar from
  the sibling npz (``depth = png / 65535 * maximum_depth``). 0 is the only
  invalid encoding (MVS holes + most of the background); there is no sky
  encoding (object-centric handheld captures), so SKY_MASK is neither
  advertised nor is a ``sky_masks`` key emitted (outdoor captures can contain
  sky, so an all-False mask would be wrong GT -- same rule as MegaDepth).
* Depth (and pose translations) are in **per-sequence SfM-arbitrary units**, not
  meters (``maximum_depth`` ranges ~18 to ~18000 across sequences), hence
  ``is_metric=False``. Depth and poses are mutually consistent within a sequence.
* Intrinsics are **per-frame** (fx==fy, no skew) and vary within a sequence, so
  K is loaded from each frame's npz, in pixels of that frame's native image.
* Frame indices are non-contiguous (gaps from the selection step), so frames are
  enumerated from the split json's index lists, never ``range()``.

Splits: ``selected_seqs_train.json`` / ``selected_seqs_test.json`` cover the SAME
2511 sequences -- the split is **frame-level** (the train list is a strict subset
of the test/full list per sequence), so sequence-level train/test dedup is
impossible by construction; choose categories/sequences disjointly if you need it.

Scalability: construction reads ONLY the split json (one ~2 MB file, ~40 ms for
all 2511 sequences) -- no per-sequence directory listing or stat. All per-frame
metadata (pose, K, depth scale) lives in per-frame npz files and is read lazily,
only for the frames actually sampled. Native image sizes are read lazily from
the first frame's JPEG header per sequence and cached.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class Co3dDataset(BaseDataset):
    """CO3Dv2 (preprocessed) as a VGGT-Omega BaseDataset (video sampling, SfM-scale depth)."""

    # Depth PNGs store uint16 fractions of the per-frame `maximum_depth` scalar.
    _DEPTH_PNG_MAX = 65535.0

    # CO3D provides RGB + MVS ("geometric") depth + per-frame SfM poses/intrinsics.
    # As with the TUM/7-Scenes vendors, WORLD_POINTS / CAM_POINTS are only the
    # depth re-projected through the poses (not an independent point-cloud GT),
    # so they are NOT advertised as evaluable GT modalities -- process_one_image
    # still computes them, they just must not be scored as a point cloud.
    # No TIMESTAMP (none on disk; frame indices have selection gaps so a synthetic
    # clock would misstate inter-frame spacing) and no SKY_MASK (no sky encoding:
    # far background is simply invalid/0 depth; no "sky_masks" key is emitted
    # either -- an all-False mask for outdoor captures would be wrong GT).
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
    def co3d_frame_paths(seq_dir: str, frame_idx: int):
        """(rgb_path, depth_path, npz_path) for frame ``frame_idx`` of ``seq_dir``.

        Note the depth filename embeds the rgb name: ``frame{i:06d}.jpg.geometric.png``.
        """
        stem = f"frame{int(frame_idx):06d}"
        return (
            os.path.join(seq_dir, "images", stem + ".jpg"),
            os.path.join(seq_dir, "depths", stem + ".jpg.geometric.png"),
            os.path.join(seq_dir, "images", stem + ".npz"),
        )

    @staticmethod
    def read_co3d_frame_meta(npz_path: str):
        """Read a CO3D per-frame npz -> (c2w (4,4) float64, K (3,3) float64, max_depth float).

        Raises ValueError if a key is missing, a shape is wrong, the pose/K are
        non-finite, or ``maximum_depth`` is not a positive finite scalar.
        """
        with np.load(npz_path) as npz:
            try:
                c2w = np.asarray(npz["camera_pose"], dtype=np.float64)
                K = np.asarray(npz["camera_intrinsics"], dtype=np.float64)
                max_depth = float(npz["maximum_depth"])
            except KeyError as e:
                raise ValueError(f"CO3D npz {npz_path!r}: missing key {e}") from e
        if c2w.shape != (4, 4):
            raise ValueError(f"CO3D npz {npz_path!r}: camera_pose shape {c2w.shape} != (4, 4)")
        if K.shape != (3, 3):
            raise ValueError(f"CO3D npz {npz_path!r}: camera_intrinsics shape {K.shape} != (3, 3)")
        if not (np.isfinite(c2w).all() and np.isfinite(K).all()):
            raise ValueError(f"CO3D npz {npz_path!r}: non-finite pose/intrinsics")
        if not (np.isfinite(max_depth) and max_depth > 0):
            raise ValueError(f"CO3D npz {npz_path!r}: bad maximum_depth {max_depth}")
        return c2w, K, max_depth

    @staticmethod
    def co3d_pose_to_w2c(c2w: np.ndarray) -> np.ndarray:
        """CO3D ``camera_pose`` (camera-to-world, OpenCV axes) -> world-to-camera (3,4) float32.

        Uses the survey-verified recipe ``np.linalg.inv(camera_pose)[:3, :4]``
        (in float64). Raises ValueError on wrong shape or non-finite values.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"CO3D camera_pose: expected shape (4, 4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("CO3D camera_pose is non-finite")
        return np.linalg.inv(c2w)[:3, :4].astype(np.float32)

    @classmethod
    def read_co3d_depth(cls, path: str, max_depth: float) -> np.ndarray:
        """Read a CO3D uint16 depth PNG -> float32 (H,W) in SfM units.

        The PNG stores per-frame-normalized fractions:
        ``depth = png / 65535 * max_depth`` where ``max_depth`` is THAT frame's
        ``maximum_depth`` scalar from the sibling npz (values are not comparable
        across frames without it). 0 is the only invalid encoding and stays 0.
        """
        arr = np.asarray(Image.open(path)).astype(np.float32)
        depth = arr / cls._DEPTH_PNG_MAX * float(max_depth)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    @staticmethod
    def co3d_intrinsics(K_native, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K in pixels of the frame's native image.

        ``override``=[fx, fy, cx, cy] wins; otherwise the per-frame
        ``camera_intrinsics`` from the npz is validated and cast. Raises
        ValueError on wrong shape or non-finite values.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        K = np.asarray(K_native, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"CO3D intrinsics: expected shape (3, 3), got {K.shape}")
        if not np.isfinite(K).all():
            raise ValueError("CO3D intrinsics are non-finite")
        return K.astype(np.float32)

    @staticmethod
    def _matches(name: str, patterns) -> bool:
        """True if ``name`` ("category/seq_id") matches any glob pattern, tried
        against the full name, the category alone, and the seq_id alone."""
        category, _, seq_id = name.partition("/")
        return any(
            fnmatch.fnmatchcase(name, pat)
            or fnmatch.fnmatchcase(category, pat)
            or fnmatch.fnmatchcase(seq_id, pat)
            for pat in patterns
        )

    def __init__(
        self,
        common_conf,
        split: str = "train",
        CO3D_DIR: str = None,
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

        if CO3D_DIR is None:
            raise ValueError("CO3D_DIR must be specified")
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.CO3D_DIR = CO3D_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # Single cheap index read for ALL sequences (no per-sequence dir listing):
        # the split json maps category -> seq_id -> [frame indices on disk].
        index_path = os.path.join(CO3D_DIR, f"selected_seqs_{split}.json")
        if not os.path.isfile(index_path):
            raise ValueError(f"CO3D split index not found: {index_path}")
        with open(index_path) as f:
            index = json.load(f)

        patterns = sequences or ["*"]
        # data_store: "category/seq_id" -> sorted list of native frame indices
        # (non-contiguous; the json is the ground truth for which frames exist).
        self.data_store = {}
        for category in sorted(index):
            for seq_id in sorted(index[category]):
                name = f"{category}/{seq_id}"
                if not self._matches(name, patterns):
                    continue
                frame_indices = sorted(int(i) for i in index[category][seq_id])
                if len(frame_indices) < min_num_images:
                    logging.warning(
                        "CO3D %s: only %d frames (< %d); skipping",
                        name, len(frame_indices), min_num_images,
                    )
                    continue
                # key doubles as the inference output-dir name (category-grouped)
                self.data_store[name] = frame_indices

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable CO3D sequences under {CO3D_DIR} "
                f"(split={split!r}, sequences={patterns})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the FIRST frame of the sequence at ``local_idx``,
        read lazily from the JPEG header and cached. (CO3D frame resolution can
        vary by a pixel or two within a sequence; the first frame is
        representative -- per-frame sizes/intrinsics are handled in get_data.)"""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path, _, _ = self.co3d_frame_paths(
                os.path.join(self.CO3D_DIR, name), self.data_store[name][0]
            )
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
        frame_indices = self.data_store[seq_name]
        seq_dir = os.path.join(self.CO3D_DIR, seq_name)

        if ids is None:
            ids = np.random.choice(
                len(frame_indices), img_per_seq, replace=self.allow_duplicate_img
            )
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frame_indices), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            frame_idx = frame_indices[int(i)]
            rgb_path, depth_path, npz_path = self.co3d_frame_paths(seq_dir, frame_idx)
            image = read_image_cv2(rgb_path)
            if image is None:
                # Listed in the split json, so the file should exist; fail loudly
                # (a silent skip would yield fewer than img_per_seq frames and
                # break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"CO3D: could not read image {rgb_path} (listed in the split json but unreadable)"
                )
            c2w, K_native, max_depth = self.read_co3d_frame_meta(npz_path)
            pose_w2c = self.co3d_pose_to_w2c(c2w)
            # Intrinsics are PER FRAME in CO3D (fx varies within a sequence).
            K = self.co3d_intrinsics(K_native, self.intrinsics_override)
            depth_map = self.read_co3d_depth(depth_path, max_depth)
            if depth_map.shape != image.shape[:2]:
                raise ValueError(
                    f"CO3D: depth {depth_map.shape} != image {image.shape[:2]} for {rgb_path}"
                )
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
            "seq_name": "co3d_" + seq_name,
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
            # would be wrong GT for outdoor captures (sky folds into 0 depth).
            "original_sizes": original_sizes,
            "is_metric": False,  # per-sequence SfM-arbitrary scale
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
