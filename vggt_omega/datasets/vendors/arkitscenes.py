"""ARKitScenes vendor (DUSt3R-style preprocessed release) for the VGGT-Omega dataset API.

This is the preprocessed ARKitScenes assembly (iPad ``vga_wide`` RGB + upsampled
LiDAR ``lowres_depth`` + ARKit world-tracking poses), not the raw download::

    ARKITSCENES_DIR/
    ├── Training/                    # 4332 numeric scene dirs, but only 3344 are
    │   │                            #   valid (~988 dirs are completely EMPTY)
    │   ├── scene_list.json          # the 3344 valid scene ids — ALWAYS use this,
    │   │                            #   never glob the Training dirs
    │   ├── all_metadata.npz         # global index; 'scenes' + 'counts' (cumulative
    │   │                            #   start offsets) give cheap per-scene frame counts
    │   └── <scene_id>/
    │       ├── vga_wide/<scene_id>_<ts>.jpg      # RGB, 640x480 (landscape) OR
    │       │                                     #   480x640 (portrait), per scene
    │       ├── lowres_depth/<scene_id>_<ts>.png  # uint16 depth (mm), same HxW as RGB
    │       └── scene_metadata.npz   # images (N,), trajectories (N,4,4),
    │                                #   intrinsics (N,6), pairs (M,3)
    └── Test/                        # 24 scene dirs (only 23 valid: 41159368 is
                                     #   completely EMPTY), same per-scene layout
                                     #   but only scene_metadata.npz (no index)
                                     #   -> listdir, guarding empty dirs

Conventions used here (validated empirically against this dataset, not assumed):

* Depth is uint16 PNG in **millimeters** (meters = value / 1000); 0 = invalid.
  It is the ARKit filled/upsampled lowres depth, so frames are essentially 100%
  dense. Cross-frame reprojection confirms the /1000 scale. Indoor: no sky, so
  SKY_MASK is advertised like the other sky-free indoor vendors (TUM/7-Scenes)
  and the emitted ``sky_masks`` are always all-False (sky convention = depth<0).
* ``trajectories`` are **camera-to-world in the OpenCV optical frame** (x-right,
  y-down, z-forward); world->camera is ``[R^T | -R^T t]`` (rotations are
  orthonormal to ~1e-15, so the transpose form is exact). Cross-frame depth
  reprojection closes to 0.4–1.2% median relative depth error across landscape,
  portrait and Test scenes, confirming pose convention x depth scale x intrinsics.
* Intrinsics are **per-frame**: ``intrinsics`` is (N,6) = [w, h, fx, fy, cx, cy]
  (ARKit per-frame calibration; fx drifts up to ~17 px within a scene), so one K
  is assembled per frame — never a single per-scene K. The stored (w, h) always
  matches the actual image resolution (mixed landscape/portrait across scenes,
  fixed within a scene).
* metadata ``images`` entries end in ``.png`` (the depth filename); the RGB file
  is the same stem with ``.jpg`` under ``vga_wide/``. Timestamps are the filename
  suffix in seconds (~10 Hz capture).
* The npz arrays are **NOT time-sorted on disk** (verified; an earlier survey's
  claim that they come pre-sorted is wrong). Frames are sorted by timestamp at
  load, applying the same permutation to trajectories/intrinsics, so video
  sampling and ``get_nearby_ids`` see a temporally ordered sequence.

Scene metadata is loaded **lazily** (one ``scene_metadata.npz`` per scene, cached
on first access): the Training split has 3344 scenes / ~957k frames, and eagerly
parsing every per-scene npz at construction would dominate startup on a network
FS. Construction only reads ``scene_list.json`` + the (lazily indexed)
``all_metadata.npz`` 'scenes'/'counts' members (Training) or one ``listdir`` per
scene (Test) to apply the ``min_num_images`` pre-filter cheaply; if the global
index is missing the pre-filter is skipped with a warning (frame counts from
``sequence_num_frames`` remain authoritative via the lazy per-scene load).
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os
import random
import zipfile

import cv2
import numpy as np

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class ArkitScenesDataset(BaseDataset):
    """ARKitScenes (preprocessed) as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # ARKitScenes provides RGB + metric LiDAR depth + GT ARKit poses/per-frame
    # intrinsics. As with the TUM vendor, WORLD_POINTS / CAM_POINTS are only the
    # depth re-projected through the GT poses (not an independent point-cloud GT),
    # so they are NOT advertised as evaluable GT modalities -- process_one_image
    # still computes them (e.g. for depth-supervised point heads), they just must
    # not be scored as a point cloud. SKY_MASK is advertised like the other
    # sky-free indoor vendors (TUM/7-Scenes): the data is indoor, so the emitted
    # sky_masks are always all-False (sky convention = depth<0).
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.TIMESTAMP,
            Modality.POINT_MASK,
            Modality.SKY_MASK,
        }
    )

    # Split name -> on-disk split directory.
    _SPLIT_DIRS = {"train": "Training", "test": "Test"}

    @staticmethod
    def arkit_pose_to_w2c(pose_c2w) -> np.ndarray:
        """ARKitScenes (4,4) camera-to-world (OpenCV axes) -> world-to-camera (3,4) float32.

        The pose is a rigid [R|t] camera-to-world, so world->camera is
        [R^T | -R^T t] (exact; ARKit rotations are orthonormal to ~1e-15, so no
        matrix inverse is needed). Raises ValueError on a malformed or non-finite
        pose (defensive: the on-disk poses were verified all-finite).
        """
        pose_c2w = np.asarray(pose_c2w, dtype=np.float64)
        if pose_c2w.shape != (4, 4):
            raise ValueError(
                f"ARKitScenes pose: expected a (4,4) matrix, got shape {pose_c2w.shape}"
            )
        if not np.isfinite(pose_c2w).all():
            raise ValueError("ARKitScenes pose is non-finite")
        rot_c2w = pose_c2w[:3, :3]
        trans_c2w = pose_c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_arkit_depth(path: str, depth_scale: float = 1000.0) -> np.ndarray:
        """Read an ARKitScenes uint16 depth PNG -> float32 (H,W) meters.

        Depth is stored as plain uint16 millimeter counts (meters = value /
        depth_scale); 0 stays 0 (invalid). Raises FileNotFoundError if the file
        is missing or unreadable (a silent skip would break fixed-V stacking).
        """
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"ARKitScenes: could not read depth {path}")
        depth = arr.astype(np.float32) / float(depth_scale)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    @staticmethod
    def arkit_intrinsics(row, override=None) -> np.ndarray:
        """(3,3) pinhole K from one ARKitScenes per-frame intrinsics row.

        ``row`` is the scene_metadata 'intrinsics' entry [w, h, fx, fy, cx, cy]
        (per-frame ARKit calibration). ``override``=[fx, fy, cx, cy] wins; else
        raises ValueError if the row is not 6 values.
        """
        if override is not None:
            fx, fy, cx, cy = override
        else:
            row = np.asarray(row, dtype=np.float64).reshape(-1)
            if row.size != 6:
                raise ValueError(
                    f"ARKitScenes intrinsics row must be [w,h,fx,fy,cx,cy]; got {row.size} values"
                )
            fx, fy, cx, cy = row[2], row[3], row[4], row[5]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def parse_arkit_timestamp(name: str) -> float:
        """Capture timestamp (seconds) from an ARKitScenes frame filename.

        Filenames are ``<scene_id>_<timestamp>.png|.jpg`` (e.g.
        ``40753679_6790.148.png`` -> 6790.148). Raises ValueError if unparsable.
        """
        stem = os.path.splitext(os.path.basename(str(name)))[0]
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            raise ValueError(f"ARKitScenes frame name {name!r}: no '_<timestamp>' suffix")
        try:
            return float(parts[1])
        except ValueError as e:
            raise ValueError(f"ARKitScenes frame name {name!r}: bad timestamp {parts[1]!r}") from e

    @staticmethod
    def _training_scene_counts(split_dir: str) -> dict:
        """Cheap {scene_id: frame_count} for the Training split from
        ``all_metadata.npz`` ('counts' holds cumulative start offsets into the
        flat global arrays; per-scene counts are the diffs). npz members are
        lazily indexed, so only the tiny 'scenes'/'counts'/'sceneids' arrays are
        read -- not the ~190MB of global poses/intrinsics. Returns {} (with a
        warning) if the index is missing, which only disables the advisory
        ``min_num_images`` pre-filter.
        """
        path = os.path.join(split_dir, "all_metadata.npz")
        try:
            index = np.load(path, allow_pickle=True)
            scenes = index["scenes"]
            offsets = np.asarray(index["counts"], dtype=np.int64)
            total = int(index["sceneids"].shape[0])
            per_scene = np.diff(np.append(offsets, total))
            return {str(s): int(c) for s, c in zip(scenes, per_scene)}
        except (OSError, KeyError, zipfile.BadZipFile) as e:
            logging.warning(
                "ARKitScenes: cannot read per-scene counts from %s (%s); "
                "min_num_images pre-filter skipped", path, e,
            )
            return {}

    def _load_scene(self, seq_name: str) -> list:
        """Lazily load one scene's ``scene_metadata.npz`` -> cached, time-sorted
        frame records ``(rgb_path, depth_path, pose_c2w (4,4) float64,
        intrinsics_row (6,), timestamp float)``.

        The npz arrays are not time-sorted on disk, so frames are argsorted by
        the filename timestamp with the SAME permutation applied to
        trajectories/intrinsics (the three arrays are row-aligned).
        """
        frames = self._scene_cache.get(seq_name)
        if frames is not None:
            return frames
        scene_dir = self.data_store[seq_name]
        meta_path = os.path.join(scene_dir, "scene_metadata.npz")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"ARKitScenes scene {seq_name}: missing {meta_path}")
        meta = np.load(meta_path, allow_pickle=True)
        names = meta["images"]
        trajectories = meta["trajectories"]
        intrinsics = meta["intrinsics"]
        if not (len(names) == len(trajectories) == len(intrinsics)):
            raise ValueError(
                f"ARKitScenes scene {seq_name}: misaligned metadata "
                f"(images {len(names)}, trajectories {len(trajectories)}, "
                f"intrinsics {len(intrinsics)})"
            )
        timestamps = np.array([self.parse_arkit_timestamp(n) for n in names])
        order = np.argsort(timestamps, kind="stable")
        frames = []
        for i in order:
            depth_name = str(names[i])  # metadata names end .png = the depth file
            rgb_name = os.path.splitext(depth_name)[0] + ".jpg"
            frames.append(
                (
                    os.path.join(scene_dir, "vga_wide", rgb_name),
                    os.path.join(scene_dir, "lowres_depth", depth_name),
                    trajectories[i],
                    intrinsics[i],
                    float(timestamps[i]),
                )
            )
        self._scene_cache[seq_name] = frames
        # (w, h) from the first frame's intrinsics row; orientation is fixed
        # within a scene and always matches the actual image resolution.
        first_row = intrinsics[order[0]]
        self._native_size_cache[seq_name] = (int(first_row[1]), int(first_row[0]))
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        ARKITSCENES_DIR: str = None,
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

        if ARKITSCENES_DIR is None:
            raise ValueError("ARKITSCENES_DIR must be specified")
        if split not in self._SPLIT_DIRS:
            raise ValueError(f"split must be one of {sorted(self._SPLIT_DIRS)}, got {split!r}")

        self.ARKITSCENES_DIR = ARKITSCENES_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.depth_scale = depth_scale
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        split_dir = os.path.join(ARKITSCENES_DIR, self._SPLIT_DIRS[split])
        if not os.path.isdir(split_dir):
            raise ValueError(f"ARKitScenes split dir not found: {split_dir}")

        # Enumerate scene NAMES cheaply. Training: scene_list.json (REQUIRED --
        # ~988 of the 4332 Training dirs are completely empty, so globbing dirs
        # is wrong). Test: no index file ships, so one directory listing.
        if split == "train":
            scene_list_path = os.path.join(split_dir, "scene_list.json")
            with open(scene_list_path) as f:
                names = [str(s) for s in json.load(f)]
        else:
            names = [
                d for d in os.listdir(split_dir)
                if os.path.isdir(os.path.join(split_dir, d))
            ]

        patterns = sequences or ["*"]
        selected = sorted(
            n for n in names if any(fnmatch.fnmatch(n, p) for p in patterns)
        )

        # Advisory per-scene frame counts for the min_num_images pre-filter,
        # WITHOUT loading any per-scene npz: Training reads the tiny
        # 'scenes'/'counts' members of all_metadata.npz; Test does one listdir
        # of lowres_depth per selected scene (24 scenes max).
        if split == "train":
            counts = self._training_scene_counts(split_dir)
        else:
            # The EMPTY-scene-dir quirk exists in Test too (Test/41159368 is a
            # bare dir with no frames or metadata), so a missing lowres_depth/
            # counts as 0 frames and falls to the min_num_images skip below
            # instead of crashing the listdir.
            counts = {}
            for n in selected:
                depth_dir = os.path.join(split_dir, n, "lowres_depth")
                counts[n] = len(os.listdir(depth_dir)) if os.path.isdir(depth_dir) else 0

        # data_store holds lightweight handles (scene dir paths); per-frame
        # enumeration is deferred to _load_scene (lazy, cached per scene).
        self.data_store = {}
        for name in selected:
            count = counts.get(name)
            if count is not None and count < min_num_images:
                logging.warning(
                    "ARKitScenes %s/%s: only %d frames (< %d); skipping",
                    self._SPLIT_DIRS[split], name, count, min_num_images,
                )
                continue
            self.data_store[name] = os.path.join(split_dir, name)

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable ARKitScenes sequences under {split_dir} "
                f"(split={split!r}, sequences={patterns})"
            )
        self._scene_cache = {}
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration). Lazily loads
        (and caches) that one scene's metadata."""
        return len(self._load_scene(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the scene's per-frame intrinsics
        (the stored (w, h) always matches the image resolution) and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            self._load_scene(name)
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
        frames = self._load_scene(seq_name)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, timestamps, original_sizes = [], [], []

        for i in ids:
            rgb_path, depth_path, pose_c2w, intri_row, ts = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Listed in scene_metadata.npz, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"ARKitScenes: could not read image {rgb_path} "
                    "(listed in scene_metadata.npz but unreadable)"
                )
            depth_map = self.read_arkit_depth(depth_path, self.depth_scale)
            pose_w2c = self.arkit_pose_to_w2c(pose_c2w)
            # Intrinsics are PER-FRAME (ARKit calibration drifts within a scene).
            K = self.arkit_intrinsics(intri_row, self.intrinsics_override)
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
            sky_masks.append(depth_map < 0)  # ARKitScenes is indoor: always all-False (sky convention = depth<0)
            timestamps.append(ts)
            original_sizes.append(original_size)

        return {
            "seq_name": "arkitscenes_" + seq_name,
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
            "timestamps": np.array(timestamps, dtype=np.float64),
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
