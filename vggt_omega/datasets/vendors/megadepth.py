"""MegaDepth vendor for the VGGT-Omega dataset API.

This loads a processed MegaDepth copy (DUSt3R-preprocessing style: per-frame EXR
depth + safetensor cameras, NOT the official h5py distribution) laid out as::

    {MEGADEPTH_DIR}/{SCENE}/{SUB}/{stem}.jpg          RGB (JPEG)
    {MEGADEPTH_DIR}/{SCENE}/{SUB}/{stem}.exr          float32 (H,W) MVS depth
    {MEGADEPTH_DIR}/{SCENE}/{SUB}/{stem}.safetensor   'cam2world' (4,4) + 'intrinsics' (3,3)

with ~175 four-digit scene dirs each holding 1-2 sub-reconstruction dirs
(``0/``, ``1/``) -- 210 scene-sub dirs / ~119k frames total. Each ``SCENE/SUB``
is one independent COLMAP sub-reconstruction and is treated as one sequence;
sub-dirs ``0/`` and ``1/`` of the same scene live in DIFFERENT world frames and
scales and must never be mixed. File stems embed the original photo extension
and can be uppercase (``x.jpg.jpg``, ``P1010264.JPG.jpg``); frames are
enumerated by matching ``*.safetensor`` and stripping that suffix.

Conventions used here (validated empirically against this dataset, not assumed):

* Depth EXRs are single-channel float32 in **SfM scale, not meters** (scale is
  arbitrary and different per sub-reconstruction); 0 encodes invalid INCLUDING
  sky (no separate sky label exists, so SKY_MASK is neither advertised nor is a
  ``sky_masks`` key emitted -- these are outdoor photos, so an all-False mask
  would be wrong GT, unlike the genuinely sky-free indoor TUM/7-Scenes).
  Zero-fraction is commonly 0.1-0.9.
* QUIRK: many frames carry degenerate "ordinal" depth where every valid pixel
  is one constant (typically 2.0; whole scenes like 0323 are mostly such
  frames). ``read_megadepth_depth`` zeroes the depth of any frame with fewer
  than ``min_depth_unique`` distinct positive values, so those frames
  contribute image+pose but no depth/point supervision (their ``point_masks``
  are all-False). Skipping them instead would break fixed-V batch stacking.
* ``cam2world`` is camera-to-world in the OpenCV optical frame (x-right,
  y-down, z-forward); world->camera is its rigid inverse with no axis flip.
  Cross-frame depth reprojection closes to ~0.1% median relative error on
  sequential DSLR scenes, confirming the convention.
* ``intrinsics`` is a per-frame (3,3) pinhole K in pixels of that frame's
  native image: fx == fy, principal point exactly centered (cx = (W-1)/2,
  cy = (H-1)/2; images are undistorted/recentered). Focal length VARIES PER
  FRAME within Flickr scenes, so K is read per frame, never per sequence.
* Resolution varies per frame (short side fixed at 600, long side ~800-1100,
  both landscape and portrait); ``native_image_size`` reports the FIRST frame
  of a sequence. The depth EXR shape always matches its JPG.

MegaDepth is an unordered internet-photo collection: ``is_video=False``,
``is_metric=False``, no timestamps. File order is alphabetical, not temporal,
so index-window sampling (``get_nearby``) only approximates covisibility on
sequential DSLR scenes; covisibility-aware sampling via the root
``megadepth_sets_64.npz`` is possible future work. That npz is NOT used for
enumeration: it indexes 253 scene-subs of which 43 are missing on disk (and
lists a filtered image subset for some scenes), so sequences are enumerated
from the directory tree instead.

Frames are enumerated **lazily** (per sequence, on first access): construction
only lists the scene/sub directory NAMES (two levels of ``scandir``), so it
finishes in a few seconds even on a network FS. A consequence is that
``min_num_images`` cannot drop sequences from ``sequence_list`` at
construction; a too-small sequence is warned about at first access instead
(never observed: the on-disk minimum is 74 frames).
"""
from __future__ import annotations

import fnmatch
import logging
import os
import random

import numpy as np

# EXR decoding requires this env var BEFORE cv2 is first imported.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
from PIL import Image
from safetensors import safe_open

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class MegaDepthDataset(BaseDataset):
    """MegaDepth as a VGGT-Omega BaseDataset (unordered multi-view, SfM-scale depth)."""

    # Camera/frame files share a stem; frames are keyed by the safetensor.
    _CAMERA_SUFFIX = ".safetensor"
    _RGB_SUFFIX = ".jpg"
    _DEPTH_SUFFIX = ".exr"

    # MegaDepth provides RGB + semi-dense MVS depth + per-frame SfM poses and
    # intrinsics. As with the TUM vendor, WORLD_POINTS / CAM_POINTS are only the
    # depth re-projected through the SfM poses (not an independent point-cloud
    # GT), so they are NOT advertised as evaluable GT modalities --
    # process_one_image still computes them, they just must not be scored as a
    # point cloud. No sky labels (sky is folded into depth==0 invalid) -- and
    # since these outdoor photos DO contain sky, an all-False ``sky_masks``
    # would be wrong GT, so the key is not emitted at all (same
    # not-advertised/not-fabricated rule as timestamps; carry_extra_modalities
    # would otherwise tensorize it for key-checking consumers). No timestamps,
    # no camera ids.
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
    def megadepth_c2w_to_w2c(c2w) -> np.ndarray:
        """(4,4) camera-to-world (OpenCV axes) -> world-to-camera (3,4) float32.

        The pose is a rigid [R|t], so world->camera is [R^T | -R^T t] (exact; no
        matrix inverse needed). Raises ValueError on a wrong shape or non-finite
        values.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"MegaDepth cam2world: expected (4,4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("MegaDepth cam2world is non-finite")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @classmethod
    def read_megadepth_camera(cls, path: str):
        """Read a frame's ``.safetensor`` -> (w2c (3,4) float32, K (3,3) float32).

        The file stores 'cam2world' (4,4 float64, camera-to-world, OpenCV axes --
        verified by cross-frame depth reprojection) and 'intrinsics' (3,3 float64
        pinhole K in pixels of the frame's native image).
        """
        with safe_open(path, framework="np") as f:
            c2w = f.get_tensor("cam2world")
            intri = f.get_tensor("intrinsics")
        intri = np.asarray(intri, dtype=np.float32)
        if intri.shape != (3, 3):
            raise ValueError(f"MegaDepth intrinsics {path!r}: expected (3,3), got {intri.shape}")
        if not np.isfinite(intri).all() or intri[0, 0] <= 0 or intri[1, 1] <= 0:
            raise ValueError(f"MegaDepth intrinsics {path!r} are invalid: {intri.tolist()}")
        return cls.megadepth_c2w_to_w2c(c2w), intri

    @staticmethod
    def read_megadepth_depth(path: str, min_depth_unique: int = 5) -> np.ndarray:
        """Read a MegaDepth depth EXR -> float32 (H,W) in SfM scale (NOT meters).

        0 encodes invalid (including sky); negative/non-finite values are mapped
        to 0 defensively. Degenerate "ordinal" frames -- fewer than
        ``min_depth_unique`` distinct positive values (typically every valid
        pixel == 2.0) -- are common in some scenes and carry no usable geometry,
        so their depth is zeroed entirely (pass ``min_depth_unique=0`` to
        disable the filter).
        """
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"MegaDepth: could not read depth EXR {path}")
        if depth.ndim == 3:  # defensively take the first channel (survey: always 1ch)
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        if min_depth_unique > 0:
            positive = depth[depth > 0]
            if positive.size and np.unique(positive).size < min_depth_unique:
                depth = np.zeros_like(depth)
        return depth

    @staticmethod
    def sequence_matches(scene: str, sub: str, patterns) -> bool:
        """Whether sequence ``scene/sub`` is selected by ``patterns``.

        A pattern containing '/' is split into scene and sub glob components
        matched separately (``"0000/0"``, ``"*/1"``); a pattern without '/'
        matches the scene name only and selects ALL its sub-reconstructions
        (``"0000"``, ``"00*"``).
        """
        for pat in patterns:
            if "/" in pat:
                scene_pat, sub_pat = pat.split("/", 1)
                if fnmatch.fnmatchcase(scene, scene_pat) and fnmatch.fnmatchcase(sub, sub_pat):
                    return True
            elif fnmatch.fnmatchcase(scene, pat):
                return True
        return False

    def __init__(
        self,
        common_conf,
        split: str = "train",
        MEGADEPTH_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        min_depth_unique: int = 5,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if MEGADEPTH_DIR is None:
            raise ValueError("MEGADEPTH_DIR must be specified")
        self.MEGADEPTH_DIR = MEGADEPTH_DIR
        self.expand_ratio = expand_ratio
        self.min_depth_unique = min_depth_unique
        self.min_num_images = min_num_images
        # No split files exist on disk (single flat collection); ``split`` only
        # selects the virtual epoch length, like the other vendors.
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        patterns = list(sequences) if sequences else ["*"]
        # Cheap two-level name enumeration only: one scandir of the root (the
        # covisibility npz and any other plain files are skipped by is_dir) and
        # one scandir per *matching* scene dir. Per-frame enumeration is
        # deferred to _frames() on first access per sequence.
        scene_pats = [pat.split("/", 1)[0] for pat in patterns]
        self.data_store = {}
        for scene_entry in sorted(os.scandir(MEGADEPTH_DIR), key=lambda e: e.name):
            if not scene_entry.is_dir():
                continue
            scene = scene_entry.name
            if not any(fnmatch.fnmatchcase(scene, sp) for sp in scene_pats):
                continue
            for sub_entry in sorted(os.scandir(scene_entry.path), key=lambda e: e.name):
                if not sub_entry.is_dir():
                    continue
                if self.sequence_matches(scene, sub_entry.name, patterns):
                    # lightweight handle: the sequence directory path
                    self.data_store[f"{scene}/{sub_entry.name}"] = sub_entry.path

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable MegaDepth sequences under {MEGADEPTH_DIR} (sequences={patterns})"
            )
        self._frames_cache = {}
        self._native_size_cache = {}

    def _frames(self, seq_name: str) -> list:
        """Frame stem paths of one sequence (lazy, cached): sorted
        ``{seq_dir}/{stem}`` prefixes such that ``stem + '.jpg'/'.exr'/'.safetensor'``
        are the frame's three files. Listed on first access only."""
        frames = self._frames_cache.get(seq_name)
        if frames is None:
            seq_dir = self.data_store[seq_name]
            n = len(self._CAMERA_SUFFIX)
            frames = sorted(
                os.path.join(seq_dir, entry.name[:-n])
                for entry in os.scandir(seq_dir)
                if entry.name.endswith(self._CAMERA_SUFFIX)
            )
            if len(frames) < self.min_num_images:
                # Frames are enumerated lazily, so unlike eager vendors this
                # cannot drop the sequence from sequence_list; warn loudly
                # instead (on-disk minimum is 74 frames, so never expected).
                logging.warning(
                    "MegaDepth seq %s: only %d frames (< %d)",
                    seq_name, len(frames), self.min_num_images,
                )
            self._frames_cache[seq_name] = frames
        return frames

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration). Lazily lists
        that one sequence's directory on first access."""
        return len(self._frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the FIRST frame of the sequence at ``local_idx``,
        read lazily from the JPEG header and cached. NOTE: MegaDepth resolution
        varies per frame within a sequence (short side fixed at 600); this is
        the first frame's size, not a sequence-wide constant."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path = self._frames(name)[0] + self._RGB_SUFFIX
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
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            # NOTE: MegaDepth file order is alphabetical, not temporal; an index
            # window approximates covisibility only on sequential DSLR scenes.
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            stem = frames[int(i)]
            rgb_path = stem + self._RGB_SUFFIX
            image = read_image_cv2(rgb_path)
            if image is None:
                # Enumerated from the sequence dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(f"MegaDepth: could not read image {rgb_path}")
            depth_map = self.read_megadepth_depth(
                stem + self._DEPTH_SUFFIX, self.min_depth_unique
            )
            # Per-frame camera: focal length varies frame-to-frame within a scene.
            pose_w2c, intri = self.read_megadepth_camera(stem + self._CAMERA_SUFFIX)
            if depth_map.shape != image.shape[:2]:
                raise ValueError(
                    f"MegaDepth: depth shape {depth_map.shape} != image shape "
                    f"{image.shape[:2]} for {stem}"
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
                pose_w2c,
                intri,
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
            "seq_name": "megadepth_" + seq_name,
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
            # would be wrong GT for outdoor photos (sky lives in depth==0).
            "original_sizes": original_sizes,
            "is_metric": False,   # COLMAP/SfM scale, independent per sub-reconstruction
            "is_video": False,    # unordered internet-photo collection
            "modalities": set(self.available_modalities),
        }
