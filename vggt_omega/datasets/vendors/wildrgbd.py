"""WildRGB-D vendor for the VGGT-Omega dataset API.

WildRGB-D is a large collection of object-centric RGB-D phone captures (46
category dirs, ~2750 scenes, 100 frames each). Layout::

    {WILDRGBD_DIR}/
      selected_seqs_train.json, selected_seqs_test.json   (top-level split index:
          {category: {"scenes/scene_XXX": [frame ids]}})
      {category}/                                          (apple, TV, car, ...)
        selected_seqs_train.json, selected_seqs_test.json  (per-category copies)
        scenes/scene_XXX/
          rgb/{id:05d}.jpg          RGB JPEG, portrait ~384-386 x 512-515 (W x H)
          depth/{id:05d}.png        uint16 PNG, millimeters
          masks/{id:05d}.png        uint8 0/255 foreground OBJECT mask
          metadata/{id:05d}.npz     camera_intrinsics (3,3) f64, camera_pose (4,4) f64

Conventions used here (validated empirically against this dataset, not assumed):

* Depth is 16-bit PNG in **millimeters** (meters = value / 1000); 0 = invalid
  (~11% of pixels). Indoor/object captures: no sky encoding, so the returned
  ``sky_masks`` are the all-False ``depth < 0``. SKY_MASK is still advertised,
  matching the convention of the other indoor vendors (TUM, 7-Scenes), which
  return byte-identical all-False sky data and advertise it.
* ``camera_pose`` is **camera-to-world** in the OpenCV optical frame (x-right,
  y-down, z-forward); world->camera is its rigid inverse with no axis flip.
  Verified by cross-frame depth reprojection on wide-baseline pairs (0.32-0.48 m):
  c2w-OpenCV closes to 0.5-5% median relative depth error, the w2c and OpenGL
  hypotheses clearly lose. The first frame's pose is (numerically near) identity,
  so the world frame is the first camera frame -- poses are per-scene relative
  but **metric** scale.
* ``camera_intrinsics`` is a per-frame (3,3) pinhole K (zero skew, fx == fy) in
  pixels of the native image; constant within a scene, slightly different across
  scenes (the native size also varies slightly per scene: width 384-386 px,
  height 512-515 px). Override via ``intrinsics=[fx,fy,cx,cy]``.
* Frame filenames are non-contiguous subsampled indices of the original video
  (0, 3, 7, 10, ...) and exactly match the split-json frame lists; sorted
  numerically they remain a temporally ordered video. The dataset ships no
  per-frame timestamps, so TIMESTAMP is not advertised.
* ``masks/`` are foreground object masks (not validity, not semantics); they are
  intentionally NOT advertised as a modality.

Scalability: the top-level ``selected_seqs_{train,test}.json`` index gives every
sequence name AND its frame-id list in a single ~1 MB read, so construction does
no per-scene directory access. Per-frame metadata (.npz pose/intrinsics) is read
lazily, only for the frames actually sampled; native image sizes are read lazily
from the first frame's JPEG header and cached. Scene directories are not
stat-checked at construction (the survey verified all referenced scenes exist on
disk); a missing file fails loudly at ``get_data`` time.
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


class WildRgbdDataset(BaseDataset):
    """WildRGB-D as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # WildRGB-D provides RGB + metric phone-sensor depth + GT camera poses and
    # per-frame intrinsics. As with the TUM vendor, WORLD_POINTS / CAM_POINTS are
    # only the depth re-projected through the GT poses (not an independent
    # point-cloud GT), so they are NOT advertised as evaluable GT modalities --
    # process_one_image still computes them (e.g. for depth-supervised point
    # heads), they just must not be scored as a point cloud. No timestamps exist
    # (frame ids are subsampled video indices on an unknown clock), so TIMESTAMP
    # is not advertised. SKY_MASK IS advertised even though these indoor/object
    # captures contain no sky (sky_masks are always all-False): the other indoor
    # vendors (TUM, 7-Scenes) advertise it for the same byte-identical all-False
    # data, and downstream modality filtering must treat the vendors uniformly.
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
    def wildrgbd_pose_to_w2c(c2w) -> np.ndarray:
        """WildRGB-D ``camera_pose`` (4,4 camera-to-world, OpenCV) -> world-to-camera
        (3,4) float32.

        The pose is a rigid [R|t] camera-to-world, so world->camera is
        [R^T | -R^T t] (exact; no matrix inverse needed). Raises ValueError on a
        wrong shape or non-finite values.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"WildRGB-D pose: expected (4,4) camera-to-world, got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("WildRGB-D pose is non-finite")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_wildrgbd_depth(path: str, depth_scale: float = 1000.0) -> np.ndarray:
        """Read a WildRGB-D 16-bit depth PNG -> float32 (H,W) meters.

        Depth is stored as plain uint16 millimeter counts (meters = value /
        depth_scale). 0 is the invalid sentinel and stays 0; non-finite values
        (defensive) also map to 0.
        """
        arr = np.asarray(Image.open(path)).astype(np.float32)
        depth = arr / float(depth_scale)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    @staticmethod
    def wildrgbd_intrinsics(K=None, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K from the metadata's ``camera_intrinsics``.

        ``override``=[fx, fy, cx, cy] wins; otherwise ``K`` is validated
        ((3,3), finite, positive focals) and cast. Raises ValueError when
        neither is usable.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        if K is None:
            raise ValueError(
                "no WildRGB-D intrinsics provided; pass K or intrinsics=[fx,fy,cx,cy]"
            )
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"WildRGB-D intrinsics: expected (3,3), got {K.shape}")
        if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("WildRGB-D intrinsics are non-finite or have non-positive focals")
        return K.astype(np.float32)

    @classmethod
    def read_wildrgbd_metadata(cls, path: str, intrinsics_override=None):
        """Read a ``metadata/{id:05d}.npz`` -> (K (3,3) float32, w2c (3,4) float32).

        ``camera_intrinsics`` is the per-frame pinhole K (native pixels);
        ``camera_pose`` is camera-to-world OpenCV (see module docstring).
        """
        with np.load(path) as md:
            K_raw = md["camera_intrinsics"]
            c2w = md["camera_pose"]
        K = cls.wildrgbd_intrinsics(K_raw, override=intrinsics_override)
        w2c = cls.wildrgbd_pose_to_w2c(c2w)
        return K, w2c

    @staticmethod
    def load_split_index(root: str, split: str) -> dict:
        """Load the top-level split index -> {"{category}/{scene}": sorted [frame ids]}.

        ``split`` in {"train", "test"} reads ``selected_seqs_{split}.json``;
        "all" merges both (splits are disjoint by scene). This single read
        enumerates every sequence name AND its frame list with no per-scene
        directory access. Raises ValueError on an unknown split.
        """
        if split not in ("train", "test", "all"):
            raise ValueError(f"split must be 'train', 'test' or 'all', got {split!r}")
        out = {}
        for s in ("train", "test") if split == "all" else (split,):
            index_path = os.path.join(root, f"selected_seqs_{s}.json")
            with open(index_path) as f:
                index = json.load(f)
            for category, scenes in index.items():
                for scene_key, frame_ids in scenes.items():
                    scene = scene_key.split("/")[-1]  # "scenes/scene_XXX" -> "scene_XXX"
                    out[f"{category}/{scene}"] = sorted(int(i) for i in frame_ids)
        return out

    @staticmethod
    def _matches(name: str, patterns) -> bool:
        """True if the sequence ``name`` ("category/scene_XXX") matches any glob in
        ``patterns`` -- against the full name, the category, or the scene alone
        (so ``["apple"]`` selects all apple scenes, ``["apple/scene_002"]`` one)."""
        category, scene = name.split("/", 1)
        return any(
            fnmatch.fnmatch(name, pat)
            or fnmatch.fnmatch(category, pat)
            or fnmatch.fnmatch(scene, pat)
            for pat in patterns
        )

    def frame_paths(self, seq_name: str, frame_id: int):
        """(rgb_path, depth_path, metadata_path) for ``frame_id`` (the on-disk
        numeric id, NOT a positional index) of sequence ``seq_name``."""
        seq_dir = self._seq_dirs[seq_name]
        stem = f"{int(frame_id):05d}"
        return (
            os.path.join(seq_dir, "rgb", stem + ".jpg"),
            os.path.join(seq_dir, "depth", stem + ".png"),
            os.path.join(seq_dir, "metadata", stem + ".npz"),
        )

    def __init__(
        self,
        common_conf,
        split: str = "train",
        WILDRGBD_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        depth_scale: float = 1000.0,
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if WILDRGBD_DIR is None:
            raise ValueError("WILDRGBD_DIR must be specified")
        self.WILDRGBD_DIR = WILDRGBD_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.depth_scale = depth_scale
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        index = self.load_split_index(WILDRGBD_DIR, split)
        patterns = list(sequences) if sequences else None

        self.data_store = {}
        self._seq_dirs = {}
        for name in sorted(index):
            if patterns is not None and not self._matches(name, patterns):
                continue
            frame_ids = index[name]
            if len(frame_ids) < min_num_images:
                logging.warning(
                    "WildRGB-D seq %s: only %d frames (< %d); skipping",
                    name, len(frame_ids), min_num_images,
                )
                continue
            category, scene = name.split("/", 1)
            # key doubles as the inference output-dir name (category-grouped)
            self.data_store[name] = frame_ids
            self._seq_dirs[name] = os.path.join(WILDRGBD_DIR, category, "scenes", scene)

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable WildRGB-D sequences under {WILDRGBD_DIR} "
                f"(split={split!r}, sequences={patterns})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached.
        (Size varies slightly across scenes: width 384-386 px, height 512-515 px.)"""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path, _, _ = self.frame_paths(name, self.data_store[name][0])
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
        sky_masks, original_sizes = [], []

        for i in ids:
            rgb_path, depth_path, meta_path = self.frame_paths(seq_name, frames[int(i)])
            image = read_image_cv2(rgb_path)
            if image is None:
                # Listed in the split index, so the file should exist; fail loudly
                # (a silent skip would yield fewer than img_per_seq frames and
                # break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"WildRGB-D: could not read image {rgb_path} (listed in the split index)"
                )
            depth_map = self.read_wildrgbd_depth(depth_path, self.depth_scale)
            K, pose_w2c = self.read_wildrgbd_metadata(meta_path, self.intrinsics_override)
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
            sky_masks.append(depth_map < 0)  # indoor/object captures: always all-False (sky convention = depth<0)
            original_sizes.append(original_size)

        return {
            "seq_name": "wildrgbd_" + seq_name,
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
