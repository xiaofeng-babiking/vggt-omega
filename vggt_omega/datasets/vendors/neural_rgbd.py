"""Neural RGB-D (Azinovic et al., CVPR 2022) vendor for the VGGT-Omega dataset API.

Neural RGB-D Surface Reconstruction is a synthetic indoor benchmark of 9 scenes
(breakfast_room, complete_kitchen, green_room, grey_white_room, kitchen,
morning_apartment, staircase, thin_geometry, whiteroom), each laid out as::

    <scene>/images/img{i}.png            RGB, 640x480, 8-bit PNG
    <scene>/depth/depth{i}.png           GT rendered depth, uint16 millimeters
    <scene>/depth_filtered/depth{i}.png  filtered noisy depth (heavy zero-dropout)
    <scene>/depth_with_noise/depth{i}.png simulated sensor depth (heavy zero-dropout)
    <scene>/focal.txt                    single float focal length (px, fx == fy)
    <scene>/poses.txt                    GT trajectory, 4x4 per frame (4 text rows)
    <scene>/trainval_poses.txt           estimated trajectory (noisy, has NaN frames)

Conventions used here (validated empirically against this dataset, not assumed):

* Depth is 16-bit PNG in **millimeters** (meters = value / 1000, verified jointly
  with the pose translations by cross-frame reprojection, median rel err ~4e-4).
  0 is the invalid sentinel and maps to 0 (GT depth has zeros where no geometry
  was rendered, e.g. thin_geometry/whiteroom). No 65535 sentinel observed.
* ``poses.txt`` holds camera-to-world matrices with **OpenGL camera axes**
  (camera looks along -Z, +Y up). World->camera OpenCV requires negating the
  Y and Z camera axes (columns 1, 2 of the rotation) before inverting:
  reprojection closes at ~6e-4 median relative error with the flip vs ~0.03+
  (or no overlap at all) without it. Do NOT treat the matrices as OpenCV c2w.
* ``trainval_poses.txt`` is the *estimated* trajectory the original paper
  optimizes from; it has NaN-marked invalid frames (Windows ``-nan(ind)``
  literals) and slightly worse closure, so this vendor ignores it and always
  reads the GT ``poses.txt`` (which is NaN-free in all 9 scenes).
* Intrinsics: ``focal.txt`` stores fx == fy (554.2562584220408 for all 9
  scenes); cx, cy are not stored anywhere and are ((W-1)/2, (H-1)/2) =
  (319.5, 239.5), verified via reprojection. Override with
  ``intrinsics=[fx, fy, cx, cy]`` if you have a different calibration.
* Frame indices in filenames are NOT zero-padded (img0.png ... img1676.png);
  frames are sorted numerically. Indices are contiguous 0..N-1 and aligned
  positionally with the pose rows (counts are validated at pose load).
* thin_geometry additionally ships a ``depth_gt/`` dir (hole-filled GT, no
  zeros) absent from the other 8 scenes; it is not uniform so not exposed.

No split files exist on disk: the whole dataset is an evaluation benchmark, so
``split`` only selects the virtual epoch length (len_train vs len_test).
No per-frame timestamps or frame-rate documentation exist either, so TIMESTAMP
is not advertised. Scenes are indoor and sky-free, so SKY_MASK is advertised
with all-False masks (TUM/7-Scenes convention).

Pose files are read **lazily** (first sample from a scene), parsed in one shot
and cached as a (N,3,4) world->camera array; construction only globs the RGB
frames and reads the tiny per-scene ``focal.txt``.
"""
from __future__ import annotations

import glob
import logging
import os
import random
import re

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class NeuralRgbdDataset(BaseDataset):
    """Neural RGB-D as a VGGT-Omega BaseDataset (video sampling, metric depth)."""

    # Column flip mapping OpenGL camera axes (x-right, y-up, z-backward) to
    # OpenCV (x-right, y-down, z-forward): negate the Y and Z camera axes.
    _GL_TO_CV_FLIP = np.diag([1.0, -1.0, -1.0])

    _DEPTH_VARIANTS = ("depth", "depth_filtered", "depth_with_noise")

    # Neural RGB-D provides RGB + GT rendered metric depth + GT camera poses.
    # As with the TUM/7-Scenes vendors, WORLD_POINTS / CAM_POINTS are only the
    # depth re-projected through the GT poses (not an independent point-cloud
    # GT such as a laser scan), so they are NOT advertised as evaluable GT
    # modalities -- process_one_image still computes them (e.g. for
    # depth-supervised point heads), they just must not be scored as a point
    # cloud. Indoor + sky-free => SKY_MASK advertised, always all-False.
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
    def read_neural_rgbd_poses(path: str) -> np.ndarray:
        """Read a Neural RGB-D pose file -> (N,4,4) float64 camera-to-world
        (OpenGL camera axes), one 4x4 matrix per frame as 4 text rows.

        NaN-tolerant: Windows-style ``-nan(ind)`` / ``nan(ind)`` literals (seen
        in ``trainval_poses.txt``) are mapped to NaN instead of crashing the
        parse; the GT ``poses.txt`` files are NaN-free. Uses a plain float split
        rather than ``np.loadtxt`` (much faster per file).
        """
        with open(path) as f:
            txt = f.read().replace("-nan(ind)", "nan").replace("nan(ind)", "nan")
        vals = np.asarray(txt.split(), dtype=np.float64)
        if vals.size == 0 or vals.size % 16 != 0:
            raise ValueError(
                f"Neural RGB-D poses {path!r}: expected a multiple of 16 values, got {vals.size}"
            )
        return vals.reshape(-1, 4, 4)

    @classmethod
    def opengl_c2w_to_w2c(cls, c2w_gl) -> np.ndarray:
        """(4,4) camera-to-world with OpenGL camera axes -> world-to-camera
        (3,4) float32 OpenCV.

        Negates the Y/Z camera axes (rotation columns 1, 2; the survey-verified
        ``c2w_cv = c2w_gl @ diag(1,-1,-1,1)`` flip), then inverts the rigid pose
        exactly via [R^T | -R^T t]. Raises ValueError on a non-(4,4) input or a
        non-finite pose (NaN frames exist in ``trainval_poses.txt``).
        """
        c2w = np.asarray(c2w_gl, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"Neural RGB-D pose: expected (4,4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("Neural RGB-D pose is non-finite (invalid/failed frame)")
        rot_c2w = c2w[:3, :3] @ cls._GL_TO_CV_FLIP  # OpenGL -> OpenCV camera axes
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_neural_rgbd_depth(path: str, depth_scale: float = 1000.0) -> np.ndarray:
        """Read a Neural RGB-D 16-bit depth PNG -> float32 (H,W) meters.

        Depth is stored as plain uint16 millimeter counts (meters = value /
        depth_scale); 0 is the invalid sentinel and stays 0.
        """
        arr = np.asarray(Image.open(path)).astype(np.float32)
        depth = arr / float(depth_scale)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    @staticmethod
    def neural_rgbd_intrinsics(focal: float, image_hw=(480, 640), override=None) -> np.ndarray:
        """(3,3) pinhole K for a Neural RGB-D scene.

        ``override``=[fx, fy, cx, cy] wins; otherwise fx = fy = ``focal`` (the
        scene's ``focal.txt``) and the principal point is the verified
        ((W-1)/2, (H-1)/2) of the native ``image_hw`` -- (319.5, 239.5) at the
        dataset's 640x480.
        """
        if override is not None:
            fx, fy, cx, cy = override
        else:
            fx = fy = float(focal)
            h, w = image_hw
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def _list_frames(scene_dir: str, depth_variant: str) -> list:
        """List a scene dir -> [(rgb_path, depth_path, frame_idx)] sorted by
        frame index. Indices are parsed numerically (filenames are NOT
        zero-padded: img0.png ... img1676.png). Poses are NOT read here (lazy).
        """
        frames = []
        for rgb_path in glob.glob(os.path.join(scene_dir, "images", "img*.png")):
            m = re.fullmatch(r"img(\d+)\.png", os.path.basename(rgb_path))
            if m is None:
                continue
            idx = int(m.group(1))
            depth_path = os.path.join(scene_dir, depth_variant, f"depth{idx}.png")
            frames.append((rgb_path, depth_path, idx))
        frames.sort(key=lambda fr: fr[2])
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "test",
        NEURAL_RGBD_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        depth_scale: float = 1000.0,
        depth_variant: str = "depth",
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if NEURAL_RGBD_DIR is None:
            raise ValueError("NEURAL_RGBD_DIR must be specified")
        if depth_variant not in self._DEPTH_VARIANTS:
            raise ValueError(
                f"depth_variant must be one of {self._DEPTH_VARIANTS}, got {depth_variant!r}"
            )

        self.NEURAL_RGBD_DIR = NEURAL_RGBD_DIR
        self.split = split
        self.expand_ratio = expand_ratio
        self.depth_scale = depth_scale
        self.depth_variant = depth_variant
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # Resolve scene directories (each must hold images/, the chosen depth
        # variant, focal.txt and the GT poses.txt).
        patterns = sequences or ["*"]
        scene_dirs = sorted(
            {
                d
                for pat in patterns
                for d in glob.glob(os.path.join(NEURAL_RGBD_DIR, pat))
                if os.path.isdir(os.path.join(d, "images"))
                and os.path.isdir(os.path.join(d, depth_variant))
                and os.path.isfile(os.path.join(d, "focal.txt"))
                and os.path.isfile(os.path.join(d, "poses.txt"))
            }
        )

        self.data_store = {}
        self._focal_store = {}
        for scene_dir in scene_dirs:
            scene = os.path.basename(scene_dir.rstrip("/"))
            frames = self._list_frames(scene_dir, depth_variant)
            if len(frames) < min_num_images:
                logging.warning(
                    "Neural RGB-D %s: only %d frames (< %d); skipping",
                    scene, len(frames), min_num_images,
                )
                continue
            with open(os.path.join(scene_dir, "focal.txt")) as f:
                self._focal_store[scene] = float(f.read())
            self.data_store[scene] = frames

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable Neural RGB-D sequences under {NEURAL_RGBD_DIR} "
                f"(sequences={patterns})"
            )
        self._native_size_cache = {}
        self._pose_cache = {}

    def _load_w2c_poses(self, seq_name: str) -> np.ndarray:
        """(N,3,4) float32 world->camera OpenCV poses for ``seq_name``, parsed
        lazily from the scene's GT ``poses.txt`` and cached. Raises ValueError
        if the pose count does not match the frame count or a pose is
        non-finite (the GT files are verified NaN-free)."""
        if seq_name not in self._pose_cache:
            pose_path = os.path.join(self.NEURAL_RGBD_DIR, seq_name, "poses.txt")
            c2w_all = self.read_neural_rgbd_poses(pose_path)
            n_frames = len(self.data_store[seq_name])
            if len(c2w_all) != n_frames:
                raise ValueError(
                    f"Neural RGB-D {seq_name}: {len(c2w_all)} poses in {pose_path!r} "
                    f"but {n_frames} frames"
                )
            self._pose_cache[seq_name] = np.stack(
                [self.opengl_c2w_to_w2c(m) for m in c2w_all]
            )
        return self._pose_cache[seq_name]

    def _native_size(self, seq_name: str):
        """Native ``(H, W)`` for ``seq_name``, read lazily from the first
        frame's PNG header and cached."""
        if seq_name not in self._native_size_cache:
            rgb_path = self.data_store[seq_name][0][0]
            with Image.open(rgb_path) as im:
                w, h = im.size  # PIL reports (W, H) without decoding pixels
            self._native_size_cache[seq_name] = (h, w)
        return self._native_size_cache[seq_name]

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the sequence at ``local_idx`` of this vendor's
        ``sequence_list`` (used by ComposedDataset enumeration)."""
        return len(self.data_store[self.sequence_list[local_idx]])

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the sequence at
        ``local_idx``, read lazily from the first frame's header and cached."""
        return self._native_size(self.sequence_list[local_idx])

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
        poses_w2c = self._load_w2c_poses(seq_name)

        if ids is None:
            ids = np.random.choice(len(frames), img_per_seq, replace=self.allow_duplicate_img)
        if self.get_nearby:
            ids = self.get_nearby_ids(ids, len(frames), expand_ratio=self.expand_ratio)

        target_image_shape = self.get_target_shape(aspect_ratio)
        K = self.neural_rgbd_intrinsics(
            self._focal_store[seq_name], self._native_size(seq_name), self.intrinsics_override
        )

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, original_sizes = [], []

        for i in ids:
            rgb_path, depth_path, frame_idx = frames[int(i)]
            image = read_image_cv2(rgb_path)
            if image is None:
                # Globbed from the scene dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"Neural RGB-D: could not read image {rgb_path}"
                )
            depth_map = self.read_neural_rgbd_depth(depth_path, self.depth_scale)
            pose_w2c = poses_w2c[frame_idx]
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
            sky_masks.append(depth_map < 0)  # synthetic indoor: always all-False (sky convention = depth<0)
            original_sizes.append(original_size)

        return {
            "seq_name": "neural_rgbd_" + seq_name,
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
