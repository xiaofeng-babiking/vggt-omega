"""BlendedMVS vendor for the VGGT-Omega dataset API.

BlendedMVS is an object-centric multi-view-stereo dataset rendered from
reconstructed meshes blended with the original imagery. On disk::

    <BLENDEDMVS_DIR>/
      <scene_id>/                # ~502 dirs, 24-hex-char scene ids
        00000000.jpg             # RGB, 512x384 (W x H)
        00000000.exr             # float32 single-channel z-depth, SfM units
        00000000.safetensor      # R_cam2world (3,3) f64, t_cam2world (3,) f32,
        ...                      #   intrinsics (3,3) f32; 3 files per frame,
                                 #   frame indices contiguous from 0
      new_overlap.h5             # pairwise view-overlap index (needs h5py; unused)

Conventions used here (validated empirically against this dataset, not assumed):

* Depth is z-depth in the scene's **SfM units** (per-scene arbitrary scale;
  median scene depth ranged 0.4 to 86 across samples) -- ``is_metric=False``.
  Invalid pixels are exactly 0.0 (up to ~39% of a frame). There is no sky
  encoding (unreconstructed regions are simply 0), so SKY_MASK is NOT
  advertised; ``sky_masks`` in the batch is always all-False.
* ``R_cam2world``/``t_cam2world`` are camera-to-world in the OpenCV optical
  frame (x-right, y-down, z-forward), so world->camera is ``[R^T | -R^T t]``
  with no axis flip. Cross-frame depth reprojection closes to ~0.02% median
  relative error, confirming depth, poses and intrinsics are mutually
  consistent at the per-scene SfM scale.
* Intrinsics are stored per frame (constant within a scene, varying across
  scenes) and are already in pixels of the native 512x384 frames.
* Sequences are UNORDERED multi-view collections (wide baselines between
  consecutive indices): ``is_video=False``, no timestamps. ``get_nearby``
  expands around the anchor *index*, which for BlendedMVS is not view
  proximity (the proper pair sampler would use ``new_overlap.h5``).
* No on-disk train/test split: ``split`` only selects the virtual epoch
  length (``len_train`` vs ``len_test``).

Frame enumeration is **lazy**: construction performs a single listing of the
dataset root to collect scene names; each scene directory is listed (and
cached) only on first access. Eagerly listing all ~502 scene dirs takes ~18 s
on the network FS, so ``min_num_images`` is also enforced lazily at sampling
time: sampler-chosen undersized sequences are redrawn under ``inside_random``,
an explicitly named or deterministic undersized sequence raises a clear
ValueError for ``ids=None`` access (never silently swapped for another scene),
and any sequence is still served when explicit ``ids`` are given.

EXR NOTE: OpenCV ships its OpenEXR codec disabled and only enables it when
``OPENCV_IO_ENABLE_OPENEXR`` is set. OpenCV (4.x) reads the variable lazily at
the FIRST EXR decode and then caches the verdict for the process lifetime, so
setting it below at module import works even when ``cv2`` was already imported
(e.g. by ``dataset_util``) -- but it MUST be set before the first EXR decode
attempt, because a single failed decode permanently disables the codec for the
process. Importing this module anywhere before depth loading is sufficient.

A superseded sibling packaging (``blendedmvs_previous/train``, 114-scene
subset, ``.npz`` cameras, bit-identical pixels/poses) exists next to the data
root; it is intentionally ignored.
"""
from __future__ import annotations

import os

# Must precede the first EXR decode in this process (see EXR NOTE above).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import fnmatch
import logging
import random

import cv2
import numpy as np
from PIL import Image
from safetensors import numpy as safetensors_numpy

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class BlendedMvsDataset(BaseDataset):
    """BlendedMVS as a VGGT-Omega BaseDataset (unordered multi-view, SfM-scale depth)."""

    # Expected keys/shapes of the per-frame camera safetensor.
    _CAMERA_KEYS = {
        "R_cam2world": (3, 3),
        "t_cam2world": (3,),
        "intrinsics": (3, 3),
    }

    # BlendedMVS provides RGB + rendered SfM-scale depth + GT camera poses and
    # per-frame intrinsics. As with the TUM vendor, WORLD_POINTS / CAM_POINTS
    # are only the depth re-projected through the GT poses (not an independent
    # point-cloud GT), so they are NOT advertised as evaluable GT modalities --
    # process_one_image still computes them (e.g. for depth-supervised point
    # heads), they just must not be scored as a point cloud. No sky encoding
    # exists (object-centric scenes), so SKY_MASK is not advertised either.
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
    def blendedmvs_pose_to_w2c(rot_c2w, trans_c2w) -> np.ndarray:
        """BlendedMVS (R_cam2world, t_cam2world) -> world-to-camera (3,4) float32 OpenCV.

        The stored pose is a rigid camera-to-world [R|t] already in the OpenCV
        optical frame, so world->camera is ``[R^T | -R^T t]`` (exact; no matrix
        inverse needed). Raises ValueError if the pose is non-finite.
        """
        rot_c2w = np.asarray(rot_c2w, dtype=np.float64).reshape(3, 3)
        trans_c2w = np.asarray(trans_c2w, dtype=np.float64).reshape(3)
        if not (np.isfinite(rot_c2w).all() and np.isfinite(trans_c2w).all()):
            raise ValueError("BlendedMVS pose is non-finite")
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @classmethod
    def read_blendedmvs_camera(cls, path: str):
        """Read a per-frame camera ``.safetensor`` -> (w2c (3,4) float32, K (3,3) float32).

        The file holds ``R_cam2world`` (3,3) float64, ``t_cam2world`` (3,)
        float32 and ``intrinsics`` (3,3) float32 (mixed dtypes are normalized
        here). K is already in pixels of the native 512x384 frame. Raises
        ValueError on missing keys, wrong shapes or non-finite values.
        """
        tensors = safetensors_numpy.load_file(path)
        for key, shape in cls._CAMERA_KEYS.items():
            if key not in tensors:
                raise ValueError(f"BlendedMVS camera {path!r}: missing key {key!r}")
            if tuple(tensors[key].shape) != shape:
                raise ValueError(
                    f"BlendedMVS camera {path!r}: {key} has shape "
                    f"{tuple(tensors[key].shape)}, expected {shape}"
                )
        w2c = cls.blendedmvs_pose_to_w2c(tensors["R_cam2world"], tensors["t_cam2world"])
        intri = tensors["intrinsics"].astype(np.float32)
        if not np.isfinite(intri).all():
            raise ValueError(f"BlendedMVS camera {path!r}: non-finite intrinsics")
        return w2c, intri

    @staticmethod
    def read_blendedmvs_depth(path: str) -> np.ndarray:
        """Read a BlendedMVS ``.exr`` depth map -> float32 (H,W) in SfM units.

        0 = invalid (BlendedMVS' own encoding). Non-finite and negative values
        (never observed, handled defensively) also map to 0 -- BlendedMVS has
        no sky, so nothing maps to the <0 sky convention.
        """
        try:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        except cv2.error as exc:  # codec disabled: env var came after a failed decode
            raise RuntimeError(
                "OpenCV EXR codec is disabled. OPENCV_IO_ENABLE_OPENEXR=1 must be "
                "set before the FIRST EXR decode in the process (importing "
                "vggt_omega.datasets.vendors.blendedmvs early enough does this); "
                f"a prior failed decode disables it permanently: {exc}"
            ) from exc
        if depth is None:
            raise FileNotFoundError(f"BlendedMVS: could not read depth {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0] = 0.0
        return depth

    def __init__(
        self,
        common_conf,
        split: str = "train",
        BLENDEDMVS_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if BLENDEDMVS_DIR is None:
            raise ValueError("BLENDEDMVS_DIR must be specified")

        self.BLENDEDMVS_DIR = BLENDEDMVS_DIR
        self.expand_ratio = expand_ratio
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # One cheap root listing; per-scene frame listings are deferred (see
        # module docstring). data_store holds lightweight directory handles.
        patterns = sequences or ["*"]
        scene_names = sorted(
            entry.name
            for entry in os.scandir(BLENDEDMVS_DIR)
            if entry.is_dir()
            and any(fnmatch.fnmatchcase(entry.name, pat) for pat in patterns)
        )
        self.data_store = {
            name: os.path.join(BLENDEDMVS_DIR, name) for name in scene_names
        }

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable BlendedMVS sequences under {BLENDEDMVS_DIR} "
                f"(sequences={patterns})"
            )
        self._frames_cache = {}
        self._native_size_cache = {}

    def _get_frames(self, seq_name: str) -> list:
        """Sorted frame stems (ints) for a sequence, listed lazily and cached.

        Only stems with all three of ``.jpg``/``.exr``/``.safetensor`` present
        are kept (incomplete frames are dropped with a warning).
        """
        frames = self._frames_cache.get(seq_name)
        if frames is None:
            stems = {"jpg": set(), "exr": set(), "safetensor": set()}
            for entry in os.scandir(self.data_store[seq_name]):
                stem, _, ext = entry.name.partition(".")
                if ext in stems and stem.isdigit():
                    stems[ext].add(int(stem))
            complete = stems["jpg"] & stems["exr"] & stems["safetensor"]
            dropped = len(stems["jpg"]) - len(complete)
            if dropped:
                logging.warning(
                    "BlendedMVS %s: %d frames missing depth/camera files; dropped",
                    seq_name, dropped,
                )
            frames = sorted(complete)
            self._frames_cache[seq_name] = frames
        return frames

    def _frame_paths(self, seq_name: str, stem: int):
        """(rgb_path, depth_path, camera_path) for one frame stem."""
        base = os.path.join(self.data_store[seq_name], f"{stem:08d}")
        return base + ".jpg", base + ".exr", base + ".safetensor"

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of complete frames in the sequence at ``local_idx`` of this
        vendor's ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self._get_frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        name = self.sequence_list[local_idx]
        if name not in self._native_size_cache:
            rgb_path, _, _ = self._frame_paths(name, self._get_frames(name)[0])
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
        # An explicitly named sequence must be served (or raise), never silently
        # swapped for another by the undersized-scene redraw below. Only a
        # sampler-derived seq_name (from seq_index) may be redrawn -- this
        # matches TUM/7-Scenes, where a named sequence is always served because
        # min_num_images is filtered eagerly at construction.
        seq_name_is_explicit = seq_name is not None
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

        # Frame counts are only known lazily (see module docstring), so the
        # min_num_images filter the reference vendors apply at construction is
        # enforced here for *sampled* ids; explicit ids are always served.
        if ids is None and len(frames) < self.min_num_images:
            if self.inside_random and not seq_name_is_explicit:
                for _ in range(100):
                    seq_name = self.sequence_list[
                        random.randint(0, self.sequence_list_len - 1)
                    ]
                    frames = self._get_frames(seq_name)
                    if len(frames) >= self.min_num_images:
                        break
                else:
                    raise ValueError(
                        f"BlendedMVS: could not draw a sequence with >= "
                        f"{self.min_num_images} frames in 100 attempts"
                    )
            else:
                raise ValueError(
                    f"BlendedMVS sequence {seq_name!r} has only {len(frames)} frames "
                    f"(< min_num_images={self.min_num_images}); pass explicit ids "
                    "to load it anyway"
                )

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, original_sizes = [], []

        for i in ids:
            rgb_path, depth_path, cam_path = self._frame_paths(seq_name, frames[int(i)])
            image = read_image_cv2(rgb_path)
            if image is None:
                # Listed from the sequence dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"BlendedMVS: could not read image {rgb_path}"
                )
            depth_map = self.read_blendedmvs_depth(depth_path)
            pose_w2c, K = self.read_blendedmvs_camera(cam_path)
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
            sky_masks.append(depth_map < 0)  # BlendedMVS has no sky: always all-False (sky convention = depth<0)
            original_sizes.append(original_size)

        return {
            "seq_name": "blendedmvs_" + seq_name,
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
            "is_metric": False,
            "is_video": False,
            "modalities": set(self.available_modalities),
        }
