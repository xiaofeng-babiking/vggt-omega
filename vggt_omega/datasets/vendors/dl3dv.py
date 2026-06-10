"""DL3DV vendor for the VGGT-Omega dataset API.

DL3DV(-10K subset) as re-exported on this cluster is a flat root of 6378
64-char-hex-hash scene directories, each holding a simple per-frame dump::

    <root>/<hash>/dense/rgb/frame_%05d.png   8-bit RGB PNG (1-indexed)
    <root>/<hash>/dense/cam/frame_%05d.npz   keys: 'pose', 'intrinsic'

Despite the ``dense`` directory name this is NOT a COLMAP dense workspace:
there is no depth, no sparse model, no masks/normals/semantics/timestamps --
only RGB + per-frame camera. Frames per scene range ~286-415 and are a
temporally ordered video capture.

Conventions used here (validated empirically against this dataset, not assumed):

* ``pose`` is a (4,4) float64 **camera-to-world** matrix with **OpenCV** camera
  axes (x-right, y-down, z-forward) -- the nerfstudio OpenGL conversion was
  already applied by whoever preprocessed the export. Verified by an epipolar
  (SIFT + essential-matrix Sampson error) test where the c2w-OpenCV hypothesis
  beats w2c/OpenGL alternatives by 3-6 orders of magnitude on multiple scenes.
  world->camera is therefore ``[R^T | -R^T t]`` with no axis flip.
* ``intrinsic`` is a (3,3) float64 pinhole K in pixels of the native image.
  It is stored redundantly per frame but constant within a scene; it varies
  across scenes. The principal point is exactly the image center (W/2, H/2)
  and fx ~= fy (slightly anisotropic).
* Image resolution VARIES per scene (~940-981 x 527-543, constant within a
  scene), so native geometry is always read from the data, never assumed.
* **No depth exists.** ``get_data`` returns all-zero float32 depth maps, so
  ``point_masks`` are all False and world/cam points are zeroed. Only IMAGE /
  EXTRINSICS / INTRINSICS are advertised as modalities (no
  DEPTH/POINT_MASK/SKY_MASK, and never WORLD_POINTS/CAM_POINTS). No
  ``sky_masks`` key is emitted: these outdoor-ish captures DO contain sky, so
  an all-False mask would be wrong GT (same rule as MegaDepth).
* Scale is per-scene normalized (nerfstudio-style; trajectory bbox diagonals
  cluster around ~11-13 SfM units across scenes): ``is_metric=False``. Frames
  are an ordered video: ``is_video=True``. There are no per-frame timestamps
  and no capture rate on disk, so TIMESTAMP is not advertised (we refuse to
  fabricate one without a known fps).
* There are no split files on disk; ``split`` only selects the virtual epoch
  length (``len_train`` vs ``len_test``), as in the other vendors.

Scalability: the root holds 6378 scenes on a network FS. Construction does a
single ``scandir`` of the root for sequence NAMES plus a threaded per-scene
frame COUNT (one ``scandir`` of each ``dense/rgb``; ~12-18s for the full root,
trivial for a small ``sequences`` filter). The count exists because short
scenes are REAL: 7 of the 6378 scenes have only 12-22 frames (an exhaustive
count; the survey's "shortest scene has 286 frames" came from a 25-scene
sample and is wrong as a global minimum). Scenes with fewer than
``min_num_images`` frames are dropped at construction with a warning -- the
TUM/7-Scenes contract -- so ``sequence_list`` is fixed up front and every
index is sampleable; raising lazily instead would crash DataLoader workers
mid-training (~7/6378 of random draws) and full-root inference enumeration
deterministically. Full frame-path enumeration (one ``glob`` of ``dense/rgb``)
and metadata stay deferred to first access per sequence and cached; camera
npz files are read lazily per sampled frame.
"""
from __future__ import annotations

import fnmatch
import glob
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class Dl3dvDataset(BaseDataset):
    """DL3DV as a VGGT-Omega BaseDataset (video sampling, RGB + cameras only)."""

    # DL3DV provides RGB + per-frame camera (pose + intrinsics) and NOTHING else:
    # no depth, so DEPTH/POINT_MASK/SKY_MASK cannot be supervised or scored, and
    # WORLD_POINTS/CAM_POINTS (which would only ever be reprojected depth, never
    # an independent point-cloud GT) are likewise not advertised. get_data still
    # returns the full core key set (all-zero depths/points, all-False masks) so
    # ComposedDataset's fixed tensorization schema keeps working; the
    # non-core "sky_masks" key is NOT emitted (an all-False mask for captures
    # that contain sky would be wrong GT).
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.EXTRINSICS,
            Modality.INTRINSICS,
        }
    )

    @staticmethod
    def dl3dv_pose_to_w2c(c2w) -> np.ndarray:
        """DL3DV ``pose`` (4,4) camera-to-world OpenCV -> world-to-camera (3,4) float32.

        The pose is a rigid [R|t] camera-to-world in the OpenCV optical frame,
        so world->camera is ``[R^T | -R^T t]`` (exact; no matrix inverse needed).
        Raises ValueError on a wrong shape or non-finite values.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"DL3DV pose: expected shape (4, 4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("DL3DV pose is non-finite")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def dl3dv_intrinsics(intrinsic, override=None) -> np.ndarray:
        """(3,3) float32 pinhole K for a DL3DV frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise ``intrinsic`` (the (3,3)
        matrix from the frame's npz) is validated and cast. Raises ValueError
        when neither is usable.
        """
        if override is not None:
            fx, fy, cx, cy = override
            return np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
            )
        if intrinsic is None:
            raise ValueError(
                "DL3DV intrinsics: no 'intrinsic' provided; pass intrinsics=[fx,fy,cx,cy]"
            )
        K = np.asarray(intrinsic, dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"DL3DV intrinsics: expected shape (3, 3), got {K.shape}")
        if not np.isfinite(K).all():
            raise ValueError("DL3DV intrinsics are non-finite")
        return K

    @classmethod
    def read_dl3dv_camera(cls, path: str, intrinsics_override=None):
        """Read a DL3DV ``cam/frame_XXXXX.npz`` -> (w2c (3,4) float32, K (3,3) float32).

        The npz holds ``pose`` (camera-to-world OpenCV, see module docstring) and
        ``intrinsic`` (pinhole K in native-image pixels). Raises ValueError when
        either key is missing or invalid.
        """
        with np.load(path) as data:
            if "pose" not in data.files or "intrinsic" not in data.files:
                raise ValueError(
                    f"DL3DV camera npz {path!r}: expected keys 'pose' and 'intrinsic', "
                    f"got {sorted(data.files)}"
                )
            c2w = data["pose"]
            intrinsic = data["intrinsic"]
        w2c = cls.dl3dv_pose_to_w2c(c2w)
        K = cls.dl3dv_intrinsics(intrinsic, override=intrinsics_override)
        return w2c, K

    @staticmethod
    def empty_depth(height: int, width: int) -> np.ndarray:
        """All-zero float32 (H,W) depth map -- DL3DV ships no depth, and in the
        DEPTH convention 0 means invalid, so every pixel is 'no measurement'
        (point_masks all False)."""
        return np.zeros((int(height), int(width)), dtype=np.float32)

    @staticmethod
    def _count_frames(seq_dir: str) -> int:
        """Number of RGB frames in a scene dir (one ``scandir`` of ``dense/rgb``,
        names only -- no stat/decode). A missing/odd ``dense/rgb`` counts as 0,
        which makes the scene fall below ``min_num_images`` and get skipped at
        construction (matching the TUM/7-Scenes warn-and-skip contract)."""
        try:
            with os.scandir(os.path.join(seq_dir, "dense", "rgb")) as it:
                return sum(
                    1
                    for e in it
                    if e.name.startswith("frame_") and e.name.endswith(".png")
                )
        except (FileNotFoundError, NotADirectoryError):
            return 0

    @staticmethod
    def _list_frames(seq_dir: str) -> list:
        """List a scene dir -> [(rgb_path, cam_path, frame_num)], ordered by
        frame number. Camera npz files are NOT read here (lazy): only the RGB
        frames are enumerated (one glob of ``dense/rgb``)."""
        frames = []
        for rgb_path in glob.glob(os.path.join(seq_dir, "dense", "rgb", "frame_*.png")):
            stem = os.path.splitext(os.path.basename(rgb_path))[0]  # "frame_00001"
            frame_num = int(stem.split("_")[1])
            cam_path = os.path.join(seq_dir, "dense", "cam", stem + ".npz")
            frames.append((rgb_path, cam_path, frame_num))
        frames.sort(key=lambda fr: fr[2])
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DL3DV_DIR: str = None,
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

        if DL3DV_DIR is None:
            raise ValueError("DL3DV_DIR must be specified")
        self.DL3DV_DIR = DL3DV_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # Enumerate sequence NAMES with a single root scandir (6378 hash dirs on
        # a network FS): per-scene frame listing / metadata is deferred to first
        # access per sequence (see _frames).
        try:
            with os.scandir(DL3DV_DIR) as it:
                all_names = sorted(
                    e.name for e in it if e.is_dir(follow_symlinks=False)
                )
        except FileNotFoundError:
            raise ValueError(f"DL3DV_DIR {DL3DV_DIR!r} does not exist")
        patterns = sequences or ["*"]
        names = sorted(
            {n for pat in patterns for n in fnmatch.filter(all_names, pat)}
        )

        # Count frames per scene (threaded scandir, names only; ~12-18s for the
        # full 6378-scene root) and DROP scenes shorter than min_num_images NOW,
        # with a warning, exactly like TUM/7-Scenes. 7 real scenes have only
        # 12-22 frames; deferring this check to first access would crash random
        # training draws and deterministic inference enumeration (see module
        # docstring). Filtering before sequence_list is fixed keeps every index
        # stable and sampleable.
        with ThreadPoolExecutor(max_workers=min(64, max(1, len(names)))) as ex:
            counts = list(
                ex.map(
                    lambda n: self._count_frames(os.path.join(DL3DV_DIR, n)), names
                )
            )
        self._frame_counts = {}
        usable = []
        for name, count in zip(names, counts):
            if count < min_num_images:
                logging.warning(
                    "DL3DV seq %s: only %d frames (< %d); skipping",
                    name, count, min_num_images,
                )
                continue
            usable.append(name)
            self._frame_counts[name] = count

        # name -> [(rgb_path, cam_path, frame_num)], filled lazily per sequence.
        self.data_store = dict.fromkeys(usable)
        self.sequence_list = usable
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable DL3DV sequences under {DL3DV_DIR} (sequences={patterns})"
            )
        self._native_size_cache = {}

    def _frames(self, seq_name: str) -> list:
        """Frame list for ``seq_name``, enumerated on first access and cached.

        Raises KeyError for unknown sequences (including scenes dropped at
        construction for having < min_num_images frames). The ValueError below
        is purely defensive: short scenes were already filtered out at
        construction, so it can only fire if the scene dir changed on disk
        between construction and first access.
        """
        if seq_name not in self.data_store:
            raise KeyError(f"DL3DV: unknown sequence {seq_name!r}")
        frames = self.data_store[seq_name]
        if frames is None:
            frames = self._list_frames(os.path.join(self.DL3DV_DIR, seq_name))
            if len(frames) < self.min_num_images:
                raise ValueError(
                    f"DL3DV seq {seq_name}: only {len(frames)} frames "
                    f"(< {self.min_num_images}) but it had "
                    f"{self._frame_counts.get(seq_name)} at construction; "
                    "the scene dir changed on disk"
                )
            self.data_store[seq_name] = frames
        return frames

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration). Served from the
        construction-time count, so enumerating all 6378 sequences stays free;
        once the lazy frame list is loaded, its length is used instead."""
        name = self.sequence_list[local_idx]
        frames = self.data_store[name]
        return self._frame_counts[name] if frames is None else len(frames)

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached.
        DL3DV resolution varies scene-to-scene, so this is always per-sequence."""
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
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        original_sizes = []

        for i in ids:
            rgb_path, cam_path, _frame_num = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Globbed from the scene dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(f"DL3DV: could not read image {rgb_path}")
            pose_w2c, K = self.read_dl3dv_camera(cam_path, self.intrinsics_override)
            original_size = np.array(image.shape[:2])
            # No depth exists for DL3DV: all-zero depth (0 = invalid everywhere).
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
            "seq_name": "dl3dv_" + seq_name,
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
            # would be wrong GT for captures that contain sky (no depth exists).
            "original_sizes": original_sizes,
            "is_metric": False,  # per-scene normalized SfM scale, not meters
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
