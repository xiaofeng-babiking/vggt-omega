"""MPI Sintel (synthetic movie benchmark) vendor for the VGGT-Omega dataset API.

Sintel is a fully synthetic film with dense GT depth and per-frame cameras.
Only the ``training/`` split exists on disk (the official 12-scene test set is
withheld by the benchmark); it holds 23 sequences of 20-50 frames each
(1064 frames total), all 1024x436 RGB::

    {SINTEL_DIR}/training/
        clean/<seq>/frame_%04d.png         render pass without effects
        final/<seq>/frame_%04d.png         render pass with motion blur/fog/effects
        albedo/<seq>/frame_%04d.png        shading-free pass (not exposed here)
        depth/<seq>/frame_%04d.dpt         float32 depth in METERS (PIEH binary)
        camdata_left/<seq>/frame_%04d.cam  K (3,3) + world->camera (3,4) (PIEH binary)

``clean`` and ``final`` share identical ``depth/`` and ``camdata_left/``; the
``render_pass`` knob selects the RGB pass (default ``"final"``, the harder,
standard robustness protocol -- e.g. MonST3R-style video-depth evals).

Conventions used here (validated empirically against this copy, not assumed):

* ``.dpt``/``.cam`` are little-endian PIEH-tagged binaries (float32 tag
  202021.25), decoded with plain numpy exactly as the on-disk
  ``sdk/python/sintel_io.py`` reference readers.
* The ``.cam`` extrinsic is stored DIRECTLY as world->camera in the OpenCV
  optical frame (x = K @ N @ X_world) -- it is used as-is, with NO inversion.
  Cross-frame depth reprojection closes to ~0.02-0.06% median relative error
  on static scenes (temple_2 f1->f30: 0.057% as w2c vs 41% if misread as c2w),
  simultaneously confirming the pose convention, metric depth and intrinsics.
* Intrinsics are stored PER FRAME and genuinely vary within a sequence for the
  zoom shots (cave_2, sleeping_1); across sequences fx = fy ranges 576..3200 px
  with cx, cy = (511.5, 217.5) = ((W-1)/2, (H-1)/2). K is therefore always read
  from the frame's ``.cam`` file, never assumed constant.
* Depth is float32 meters with no NaN/inf/zero on disk; sky/far pixels are
  encoded as LARGE FINITE depth -- either a ~1e11 sentinel (ambush_4, market_2,
  market_6, sleeping_2) or a far sky-dome continuum at ~2.5-3.4 km (ambush_2,
  mountain_1). All real (non-sky) geometry observed stays well below 1 km, so
  pixels >= ``sky_threshold`` (default 1000 m) are mapped to the repo-wide sky
  sentinel ``-1.0`` BEFORE ``process_one_image`` (DEPTH convention: 0 = invalid,
  < 0 = sky) and SKY_MASK is advertised.
* Sintel is a rendered 24 fps movie, but no per-frame clock is stored on disk
  (24 fps is film convention, not data), so TIMESTAMP is not advertised.

``min_num_images`` defaults to 8 (not the usual 24) because real Sintel
sequences are short -- ambush_6 has 20 frames and ambush_2 has 21 -- and for an
evaluation dataset silently dropping real scenes is worse than keeping short
sequences.
"""
from __future__ import annotations

import glob
import logging
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class SintelDataset(BaseDataset):
    """MPI Sintel as a VGGT-Omega BaseDataset (video sampling, metric depth,
    per-frame GT cameras, sky encoded as huge finite depth)."""

    # PIEH magic shared by Sintel's .dpt/.cam/.flo binaries (sdk/python/sintel_io.py).
    _TAG_FLOAT = 202021.25

    # The two evaluable render passes (albedo is shading-free and not exposed).
    _RENDER_PASSES = ("clean", "final")

    # Sintel provides RGB + dense metric depth + per-frame GT cameras, with sky
    # encoded as huge finite depth (mapped to the -1.0 sentinel by the depth
    # reader, so SKY_MASK is real GT). As with the TUM vendor, WORLD_POINTS /
    # CAM_POINTS are only the depth re-projected through the GT poses (not an
    # independent point-cloud GT), so they are NOT advertised as evaluable GT
    # modalities -- process_one_image still computes them, they just must not
    # be scored as a point cloud. No TIMESTAMP (no on-disk clock; 24 fps is
    # film convention, not data) and no CAMERA_ID (single left camera).
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

    @classmethod
    def _check_tag(cls, f, path: str) -> None:
        """Read and validate the leading PIEH float32 tag of a Sintel binary."""
        tag = np.fromfile(f, np.float32, 1)
        if tag.size != 1 or tag[0] != cls._TAG_FLOAT:
            raise ValueError(
                f"Sintel file {path!r}: bad or missing PIEH tag "
                f"(expected {cls._TAG_FLOAT}, got {tag})"
            )

    @classmethod
    def read_sintel_depth(cls, path: str, sky_threshold: float = 1000.0) -> np.ndarray:
        """Read a Sintel ``.dpt`` -> float32 (H,W) meters with sky as -1.0.

        Layout (little-endian, per README_depth.txt / sdk): float32 PIEH tag
        202021.25, int32 width, int32 height, then width*height float32
        row-major depth in meters. The on-disk data has no NaN/inf/zero, but
        non-finite or negative values are mapped to 0 (invalid) defensively.
        Sky/far pixels are huge finite depths (a ~1e11 sentinel or a ~2.5-3.4 km
        sky dome): pixels >= ``sky_threshold`` are mapped to the repo-wide sky
        sentinel ``-1.0`` (DEPTH convention: 0 = invalid, < 0 = sky), so
        downstream masks are ``depth > 0`` (valid) and ``depth < 0`` (sky).
        """
        with open(path, "rb") as f:
            cls._check_tag(f, path)
            dims = np.fromfile(f, np.int32, 2)
            if dims.size != 2 or (dims <= 0).any():
                raise ValueError(f"Sintel depth {path!r}: bad width/height header {dims}")
            width, height = int(dims[0]), int(dims[1])
            data = np.fromfile(f, np.float32, width * height)
        if data.size != width * height:
            raise ValueError(
                f"Sintel depth {path!r}: truncated payload "
                f"({data.size} of {width * height} float32 values)"
            )
        depth = data.reshape(height, width)
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        depth[depth >= sky_threshold] = -1.0
        return depth

    @classmethod
    def read_sintel_cam(cls, path: str):
        """Read a Sintel ``.cam`` -> (K (3,3) float32, w2c (3,4) float32).

        Layout (little-endian, per sdk): float32 PIEH tag, then the 3x3
        intrinsic K as 9 float64 row-major, then the 3x4 extrinsic as 12
        float64 row-major. The extrinsic is stored DIRECTLY as world->camera
        in the OpenCV optical frame (empirically verified via cross-frame
        reprojection closure), so it is returned as-is with NO inversion.
        Raises ValueError on a bad tag, truncation, non-finite values or
        non-positive focals.
        """
        with open(path, "rb") as f:
            cls._check_tag(f, path)
            K = np.fromfile(f, np.float64, 9)
            w2c = np.fromfile(f, np.float64, 12)
        if K.size != 9 or w2c.size != 12:
            raise ValueError(
                f"Sintel cam {path!r}: truncated (got {K.size} K + {w2c.size} extrinsic values)"
            )
        K = K.reshape(3, 3)
        w2c = w2c.reshape(3, 4)
        if not (np.isfinite(K).all() and np.isfinite(w2c).all()):
            raise ValueError(f"Sintel cam {path!r}: non-finite values")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(f"Sintel cam {path!r}: non-positive focal lengths in K")
        return K.astype(np.float32), w2c.astype(np.float32)

    def __init__(
        self,
        common_conf,
        split: str = "train",
        SINTEL_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        render_pass: str = "final",
        sky_threshold: float = 1000.0,
        min_num_images: int = 8,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if SINTEL_DIR is None:
            raise ValueError("SINTEL_DIR must be specified")
        if render_pass not in self._RENDER_PASSES:
            raise ValueError(
                f"render_pass must be one of {self._RENDER_PASSES}, got {render_pass!r}"
            )
        if sky_threshold <= 0:
            raise ValueError(f"sky_threshold must be > 0, got {sky_threshold}")

        self.SINTEL_DIR = SINTEL_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.render_pass = render_pass
        self.sky_threshold = sky_threshold
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # Only training/ exists on disk (the benchmark test set is withheld);
        # `split` selects the virtual epoch length, not an on-disk subset.
        training_dir = os.path.join(SINTEL_DIR, "training")
        if not os.path.isdir(training_dir):
            raise ValueError(
                f"Sintel training dir {training_dir} does not exist "
                "(SINTEL_DIR must be the dataset root containing training/)"
            )

        patterns = sequences or ["*"]
        seq_dirs = sorted(
            {
                d
                for pat in patterns
                for d in glob.glob(os.path.join(training_dir, render_pass, pat))
                if os.path.isdir(d)
            }
        )

        # Eager frame enumeration: the whole dataset is 23 sequences x <= 50
        # frames, so listing every sequence dir at construction is cheap.
        # Depth/cam paths are derived from the RGB stem (clean/final share
        # depth/ and camdata_left/); the binary files are only read on access.
        self.data_store = {}
        for seq_dir in seq_dirs:
            name = os.path.basename(seq_dir.rstrip("/"))
            frames = []
            for rgb_path in glob.glob(os.path.join(seq_dir, "frame_*.png")):
                stem = os.path.splitext(os.path.basename(rgb_path))[0]  # "frame_0001"
                frame_idx = int(stem.split("_")[1])
                frames.append(
                    (
                        rgb_path,
                        os.path.join(training_dir, "depth", name, stem + ".dpt"),
                        os.path.join(training_dir, "camdata_left", name, stem + ".cam"),
                        frame_idx,
                    )
                )
            frames.sort(key=lambda fr: fr[3])
            if len(frames) < min_num_images:
                logging.warning(
                    "Sintel seq %s: only %d frames (< %d); skipping",
                    name, len(frames), min_num_images,
                )
                continue
            self.data_store[name] = frames

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable Sintel sequences under {training_dir} "
                f"(render_pass={render_pass!r}, sequences={patterns})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
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
        sky_masks, original_sizes = [], []

        for i in ids:
            rgb_path, depth_path, cam_path, _frame_idx = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Enumerated from the render-pass dir, so the file should exist;
                # fail loudly (a silent skip would yield fewer than img_per_seq
                # frames and break fixed-V batch stacking).
                raise FileNotFoundError(f"Sintel: could not read image {rgb_path}")
            depth_map = self.read_sintel_depth(depth_path, self.sky_threshold)  # sky -> -1.0
            # Intrinsics genuinely vary per frame (zoom shots in cave_2 /
            # sleeping_1) and per sequence: always read K from the frame's .cam.
            K, pose_w2c = self.read_sintel_cam(cam_path)
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
            sky_masks.append(depth_map < 0)  # sky was mapped to -1.0 before processing
            original_sizes.append(original_size)

        return {
            "seq_name": "sintel_" + seq_name,
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
