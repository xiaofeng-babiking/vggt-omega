"""Hypersim vendor for the VGGT-Omega dataset API.

This loads the *preprocessed* PNG/npy/npz copy of Hypersim (Apple's
photorealistic synthetic indoor dataset), laid out as::

    HYPERSIM_DIR/ai_XXX_XXX/cam_NN/
        FFFFFF_rgb.png    8-bit tonemapped LDR RGB, 1024x768 (HDR tonemapping
                          was baked in by the preprocessor; not recoverable)
        FFFFFF_depth.npy  float32 (768,1024) planar z-depth in METERS;
                          invalid pixels are NaN (windows / missing geometry)
        FFFFFF_cam.npz    'pose' (4,4) camera-to-world, 'intrinsics' (3,3)

One sequence = one ``ai_XXX_XXX/cam_NN`` trajectory (an animated camera path
through a synthetic indoor scene; several independent ``cam_NN`` trajectories
may cover the same scene). Frame indices are 6-digit zero-padded and can be
NON-contiguous (e.g. ai_006_006/cam_02 spans 000016..000089 with gaps), so
frames are always enumerated from the files, never assumed to be ``range(N)``.

Conventions used here (validated empirically against this copy, not assumed):

* Depth is ALREADY planar z-depth in meters -- NOT Hypersim's native
  distance-to-camera-center, so no ray-to-plane conversion is applied
  (cross-frame reprojection: median rel err 3e-4 for z-depth vs 3e-2 for the
  ray-distance hypothesis). Invalid pixels are NaN (not 0/inf) and map to 0.
* ``cam.npz['pose']`` is camera-to-world in OpenCV axes (x-right, y-down,
  z-forward) -- the preprocessor already converted Hypersim's native
  OpenGL/asset-units convention. world->camera is ``np.linalg.inv(pose)[:3,:4]``
  (verified by cross-frame depth reprojection on 4 scenes, rel err <= 2.4e-3).
* ``cam.npz['intrinsics']`` is the per-frame (3,3) pixel-unit K (constant
  within a cam dir; fx=fy~883-887, principal point exactly the image center).
  Override via ``intrinsics=[fx, fy, cx, cy]`` if needed.
* SKY_MASK is NOT advertised: "sky" seen through windows appears as NaN depth,
  but NaN conflates windows with missing geometry, so it cannot be decoded
  into a trustworthy sky mask. NaN maps to depth 0 (invalid), never negative.
* No timestamps, semantics, normals or split files exist in this copy (the
  official HDF5 extras and the train/val/test scene CSV were stripped), so
  ``split`` only selects ``len_train`` vs ``len_test``; both splits see the
  same flat sequence collection.

Construction is scalable: sequence names are discovered with one directory
listing per scene (parallelized over a thread pool -- the metadata calls are
network-FS bound), and each candidate cam dir is probed with a single
early-exited ``scandir`` (36 of the 793 cam dirs are completely empty, a few
more hold fewer than ``min_num_images`` frames; both are skipped). The full
per-sequence frame list is built lazily on first access and cached, and the
per-frame ``cam.npz`` files are only read inside ``get_data``.
"""
from __future__ import annotations

import fnmatch
import glob
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class HypersimDataset(BaseDataset):
    """Hypersim (preprocessed copy) as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # Hypersim provides RGB + metric z-depth + GT poses/intrinsics. As with the
    # TUM vendor, WORLD_POINTS / CAM_POINTS are only the depth re-projected
    # through the GT poses (not an independent point-cloud GT), so they are NOT
    # advertised as evaluable GT modalities -- process_one_image still computes
    # them (e.g. for depth-supervised point heads), they just must not be
    # scored as a point cloud. SKY_MASK / TIMESTAMP are absent from this copy
    # (see module docstring).
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.POINT_MASK,
        }
    )

    _RGB_SUFFIX = "_rgb.png"

    @staticmethod
    def hypersim_pose_to_w2c(c2w) -> np.ndarray:
        """Hypersim ``cam.npz['pose']`` (4,4) camera-to-world, OpenCV axes ->
        world-to-camera (3,4) float32.

        Uses ``np.linalg.inv`` (the empirically verified recipe for this copy)
        rather than the rigid transpose trick; the two agree to ~1e-5 (the
        stored float32 rotations are orthonormal only to that level). Raises
        ValueError on a non-(4,4) or non-finite pose.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"hypersim pose: expected (4,4) camera-to-world, got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("hypersim pose is non-finite")
        return np.linalg.inv(c2w)[:3, :4].astype(np.float32)

    @staticmethod
    def read_hypersim_depth(path: str) -> np.ndarray:
        """Read a Hypersim ``*_depth.npy`` -> float32 (H,W) planar z-depth in meters.

        The stored depth is ALREADY planar z (not distance-to-camera-center),
        so no conversion is applied. Invalid pixels are stored as NaN and map
        to 0 (the registry's invalid sentinel); depth is never set negative
        because this copy has no decodable sky labels.
        """
        depth = np.asarray(np.load(path), dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"hypersim depth {path!r}: expected 2-D map, got shape {depth.shape}")
        depth[~np.isfinite(depth)] = 0.0
        return depth

    @classmethod
    def hypersim_intrinsics(cls, K_raw=None, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K from a ``cam.npz['intrinsics']`` array.

        ``override``=[fx, fy, cx, cy] wins; otherwise the stored per-frame K is
        validated and cast. Raises ValueError when neither yields a (3,3) K.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        K = np.asarray(K_raw, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(
                f"hypersim intrinsics: expected (3,3), got {K.shape}; "
                "pass intrinsics=[fx, fy, cx, cy] to override"
            )
        return K

    @classmethod
    def read_hypersim_cam(cls, path: str, intrinsics_override=None):
        """Read a ``*_cam.npz`` -> ``(K (3,3) float32, w2c (3,4) float32)``.

        Raises ValueError when the archive misses the ``pose`` / ``intrinsics``
        keys or holds an invalid pose (see :meth:`hypersim_pose_to_w2c`).
        """
        with np.load(path) as cam:
            if "pose" not in cam or "intrinsics" not in cam:
                raise ValueError(
                    f"hypersim cam file {path!r}: expected keys 'pose' and 'intrinsics', "
                    f"got {sorted(cam.keys())}"
                )
            K = cls.hypersim_intrinsics(cam["intrinsics"], intrinsics_override)
            w2c = cls.hypersim_pose_to_w2c(cam["pose"])
        return K, w2c

    @classmethod
    def _probe_scene(cls, scene_dir: str, cam_pattern: str, min_num_images: int) -> list[str]:
        """One scene dir -> kept ``scene/cam_NN`` sequence names.

        A single early-exited ``scandir`` per cam dir: counting stops as soon
        as ``min_num_images`` RGB frames are seen, and empty cam dirs cost one
        empty listing. No frame list is built here (that is lazy).
        """
        scene = os.path.basename(scene_dir.rstrip("/"))
        try:
            cams = sorted(
                e.name
                for e in os.scandir(scene_dir)
                if e.is_dir() and fnmatch.fnmatch(e.name, cam_pattern)
            )
        except NotADirectoryError:
            return []
        kept = []
        for cam in cams:
            count = 0
            with os.scandir(os.path.join(scene_dir, cam)) as it:
                for entry in it:
                    if entry.name.endswith(cls._RGB_SUFFIX):
                        count += 1
                        if count >= min_num_images:
                            break
            if count >= min_num_images:
                kept.append(f"{scene}/{cam}")
        return kept

    def __init__(
        self,
        common_conf,
        split: str = "train",
        HYPERSIM_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        intrinsics=None,
        enum_workers: int = 32,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if HYPERSIM_DIR is None:
            raise ValueError("HYPERSIM_DIR must be specified")
        self.HYPERSIM_DIR = HYPERSIM_DIR
        self.expand_ratio = expand_ratio
        self.intrinsics_override = intrinsics
        # min_num_images >= 1 always: empty cam dirs must never become sequences
        self.min_num_images = max(int(min_num_images), 1)
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # `sequences` entries match "scene", "scene/cam_NN" or glob patterns of
        # either form ("ai_001_*", "ai_001_001/cam_0[02]", ...).
        patterns = sequences or ["*"]
        tasks = []  # (scene_dir, cam_pattern), deduplicated
        seen = set()
        for pat in patterns:
            scene_pat, _, cam_pat = pat.partition("/")
            cam_pat = cam_pat or "cam_*"
            for scene_dir in glob.glob(os.path.join(HYPERSIM_DIR, scene_pat)):
                key = (scene_dir, cam_pat)
                if key not in seen:
                    seen.add(key)
                    tasks.append(key)

        # The probes are network-FS metadata calls; a thread pool keeps the
        # full 793-cam-dir enumeration to a few seconds.
        names = set()
        if tasks:
            with ThreadPoolExecutor(max_workers=min(enum_workers, len(tasks))) as pool:
                for kept in pool.map(
                    lambda t: self._probe_scene(t[0], t[1], self.min_num_images), tasks
                ):
                    names.update(kept)

        # name -> cached frame list, filled lazily by _get_frames (a lightweight
        # handle: the cam dir path is implied by the name).
        self.data_store = {name: None for name in sorted(names)}
        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable Hypersim sequences under {HYPERSIM_DIR} (sequences={patterns}, "
                f"min_num_images={self.min_num_images})"
            )
        self._native_size_cache = {}

    def _get_frames(self, seq_name: str) -> list:
        """Frame list for ``seq_name`` -> [(rgb_path, depth_path, cam_path, frame_idx)],
        ordered by frame index; built lazily (one listing) on first access and cached.

        Frame indices come from the ``FFFFFF_rgb.png`` filenames (they can be
        non-contiguous); the depth/cam paths are derived from the same stem and
        ``get_data`` fails loudly if one is missing.
        """
        frames = self.data_store[seq_name]
        if frames is None:
            seq_dir = os.path.join(self.HYPERSIM_DIR, seq_name)
            stems = []
            with os.scandir(seq_dir) as it:
                for entry in it:
                    if entry.name.endswith(self._RGB_SUFFIX):
                        stem = entry.name[: -len(self._RGB_SUFFIX)]  # "000123"
                        stems.append((int(stem), stem))
            stems.sort()
            frames = [
                (
                    os.path.join(seq_dir, stem + self._RGB_SUFFIX),
                    os.path.join(seq_dir, stem + "_depth.npy"),
                    os.path.join(seq_dir, stem + "_cam.npz"),
                    idx,
                )
                for idx, stem in stems
            ]
            self.data_store[seq_name] = frames
        return frames

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration); lazily listed."""
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

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, original_sizes = [], []

        for i in ids:
            rgb_path, depth_path, cam_path, _frame_idx = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Enumerated from the sequence dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"Hypersim: could not read image {rgb_path}"
                )
            depth_map = self.read_hypersim_depth(depth_path)
            K, pose_w2c = self.read_hypersim_cam(cam_path, self.intrinsics_override)
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
            # always all-False: Hypersim NaN "sky through windows" is mapped to
            # 0 (invalid), never to the negative sky sentinel (see docstring)
            sky_masks.append(depth_map < 0)
            original_sizes.append(original_size)

        return {
            "seq_name": "hypersim_" + seq_name,
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
