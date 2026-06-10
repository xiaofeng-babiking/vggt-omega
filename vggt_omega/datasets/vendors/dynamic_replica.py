"""Dynamic Replica vendor for the VGGT-Omega dataset API.

This loads the ``dynamic_replica_data`` re-export of Dynamic Replica: synthetic
stereo 30 fps video of Replica scenes with N animated objects (dir suffix
``-N_obj``). Layout::

    <DYNAMIC_REPLICA_DIR>/train/<6hex>-<N>_obj/{left,right}/
        rgb/<t>.png      RGBA uint8 1280x720 (alpha constant 255 -> dropped)
        depth/<t>.npy    float32 (720,1280), ALREADY METERS, 0 = invalid
        cam/<t>.npz      keys 'pose' (4,4) and 'intrinsics' (3,3)
        flow_forward/    optical-flow GT (not loaded here; no FLOW modality)
        flow_backward/

483 sequence dirs x 2 cameras = 966 camera streams; each stream has 300 frames
(10 s at 30 fps). Each camera stream (``<seq>/left``, ``<seq>/right``) is
treated as its OWN sequence; the stereo relationship is exposed through the
CAMERA_ID modality (left=0, right=1; both poses share one world frame).

Conventions used here (validated empirically against this dataset, not assumed):

* ``cam/<t>.npz['pose']`` is **camera-to-world in OpenCV axes** (x-right,
  y-down, z-forward). world->camera = the rigid inverse ``[R^T | -R^T t]``.
  Verified by a 6-hypothesis ({w2c,c2w} x {OpenCV,OpenGL,PyTorch3D}) stereo +
  temporal depth-reprojection sweep: c2w-OpenCV closes the stereo pair at
  ~0 median relative depth error and wins temporally.
* ``depth/<t>.npy`` is float32 **meters** (scale 1.0, metric). 0 (sometimes
  -0.0) marks invalid pixels (Replica mesh holes; 0-44% per scene). Indoor
  synthetic data -- there is NO sky, so the advertised SKY_MASK is always
  all-False (same indoor semantics as the TUM/7-Scenes vendors) and any
  (defensively clamped) negative value is treated as invalid, not sky.
* ``cam/<t>.npz['intrinsics']`` is a (3,3) pixel K, stored per frame but
  globally constant: fx=fy=700, cx=640, cy=360 for the native 1280x720 frame.
  The per-frame value from disk is used (override via ``intrinsics``).
* Frame filenames are float-second timestamps ``i/30`` (e.g.
  ``0.06666666666666667``). Lexicographic order scrambles temporal order once
  timestamps cross 10 s (``'10.0' < '9.96...'`` as strings; the current export
  tops out at 9.97 s but relying on that would be fragile), so frames are
  sorted by ``float(stem)`` and the stem doubles as the TIMESTAMP.

Scenes are DYNAMIC and this export ships no dynamic-object masks, so
cross-frame depth/pose consistency holds only on static pixels (fine for
nearby frames; degrades with temporal distance as objects move).

Only a ``train/`` split exists on disk (no val/test); ``split`` therefore only
selects the virtual epoch length (``len_train`` vs ``len_test``) -- frames
always come from ``train/``.

Frame lists and camera files are read **lazily** (first access per stream,
cached): the export is ~3.8 TB / ~290k frames per modality, and construction
only does ONE directory listing of ``train/`` to enumerate sequence names.
"""
from __future__ import annotations

import fnmatch
import glob
import os
import random

import numpy as np
from PIL import Image

from vggt_omega.datasets.base_dataset import BaseDataset
from vggt_omega.datasets.dataset_util import read_image_cv2
from vggt_omega.datasets.modality import Modality


class DynamicReplicaDataset(BaseDataset):
    """Dynamic Replica as a VGGT-Omega BaseDataset (stereo video, metric depth)."""

    # Survey-verified globally-constant pinhole intrinsics (fx, fy, cx, cy) for
    # the native 1280x720 frame (identical across frames/sequences/cameras).
    _FOCAL = 700.0
    _PRINCIPAL_POINT = (640.0, 360.0)

    # Stereo camera streams and their CAMERA_ID encoding.
    _CAMERA_IDS = {"left": 0, "right": 1}

    # Frame rate of the export (filename timestamps are i/30 s).
    _FPS = 30.0

    # Dynamic Replica provides RGB + metric depth + GT per-frame poses and
    # intrinsics, plus the stereo camera id and filename timestamps. As with the
    # TUM/7-Scenes vendors, WORLD_POINTS / CAM_POINTS are only the depth
    # re-projected through the GT poses (not an independent point-cloud GT), so
    # they are NOT advertised as evaluable GT modalities -- process_one_image
    # still computes them (e.g. for depth-supervised point heads), they just
    # must not be scored as a point cloud. SKY_MASK is advertised exactly like
    # the other indoor vendors (TUM/7-Scenes): the scenes are indoor synthetic
    # with no sky encoding, so the emitted masks are always all-False -- the
    # sample stays uniform across indoor vendors for consumers gating sky
    # supervision/eval on sample['modalities'].
    AVAILABLE = frozenset(
        {
            Modality.IMAGE,
            Modality.DEPTH,
            Modality.INTRINSICS,
            Modality.EXTRINSICS,
            Modality.POINT_MASK,
            Modality.SKY_MASK,
            Modality.TIMESTAMP,
            Modality.CAMERA_ID,
        }
    )

    @staticmethod
    def dynamic_replica_pose_to_w2c(c2w) -> np.ndarray:
        """(4,4) camera-to-world (OpenCV axes) -> world-to-camera (3,4) float32.

        The pose is a rigid [R|t] camera-to-world, so world->camera is
        [R^T | -R^T t] (exact; no matrix inverse needed). Raises ValueError on a
        wrong shape or non-finite values.
        """
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(
                f"Dynamic Replica pose: expected (4,4) camera-to-world, got {c2w.shape}"
            )
        if not np.isfinite(c2w).all():
            raise ValueError("Dynamic Replica pose is non-finite")
        rot_c2w = c2w[:3, :3]
        trans_c2w = c2w[:3, 3]
        rot_w2c = rot_c2w.T
        trans_w2c = -rot_w2c @ trans_c2w
        return np.concatenate([rot_w2c, trans_w2c[:, None]], axis=1).astype(np.float32)

    @staticmethod
    def read_dynamic_replica_depth(path: str) -> np.ndarray:
        """Read a Dynamic Replica ``depth/<t>.npy`` -> float32 (H,W) meters.

        Depth is stored as float32 meters (scale 1.0). 0 / -0.0 mark invalid
        pixels; non-finite values and (defensively) any negative value also map
        to 0. The dataset is indoor synthetic, so nothing is encoded as sky.
        """
        depth = np.load(path).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0] = 0.0  # negatives AND -0.0 -> +0.0; no sky here
        return depth

    @classmethod
    def read_dynamic_replica_camera(cls, path: str):
        """Read a ``cam/<t>.npz`` -> (w2c (3,4) float32, K (3,3) float32).

        The npz holds 'pose' ((4,4) camera-to-world, OpenCV axes) and
        'intrinsics' ((3,3) pixel K of the native 1280x720 frame). Raises
        ValueError if either key is missing.
        """
        with np.load(path) as npz:
            if "pose" not in npz or "intrinsics" not in npz:
                raise ValueError(
                    f"Dynamic Replica cam file {path!r}: expected keys "
                    f"'pose'/'intrinsics', got {sorted(npz.keys())}"
                )
            pose = npz["pose"]
            K = npz["intrinsics"].astype(np.float32)
        if K.shape != (3, 3):
            raise ValueError(
                f"Dynamic Replica cam file {path!r}: intrinsics shape {K.shape} != (3,3)"
            )
        return cls.dynamic_replica_pose_to_w2c(pose), K

    @classmethod
    def dynamic_replica_intrinsics(cls, override=None) -> np.ndarray:
        """(3,3) pinhole K for Dynamic Replica's native 1280x720 frame.

        ``override``=[fx, fy, cx, cy] wins; otherwise the survey-verified
        globally-constant fx=fy=700, principal point=(640, 360). (get_data uses
        the per-frame on-disk K unless ``intrinsics`` is overridden, but the
        disk value equals this constant everywhere.)
        """
        if override is not None:
            fx, fy, cx, cy = override
        else:
            fx = fy = cls._FOCAL
            cx, cy = cls._PRINCIPAL_POINT
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    @classmethod
    def camera_id_for_stream(cls, stream_name: str) -> int:
        """CAMERA_ID for a camera-stream sequence name ``<seq>/<camera>``
        (left=0, right=1). Raises ValueError for an unknown camera."""
        camera = stream_name.rsplit("/", 1)[-1]
        if camera not in cls._CAMERA_IDS:
            raise ValueError(
                f"Dynamic Replica stream {stream_name!r}: camera must be one of "
                f"{sorted(cls._CAMERA_IDS)}"
            )
        return cls._CAMERA_IDS[camera]

    @staticmethod
    def sort_frame_stems(stems) -> list:
        """Sort frame filename stems (float-second timestamps like
        ``'0.06666666666666667'``) temporally, i.e. by ``float(stem)``.

        Lexicographic order scrambles temporal order once timestamps cross
        10 s (``'10.0' < '9.96...'`` as strings), so this must be used instead
        of a plain sort. Raises ValueError if a stem is not a parseable float.
        """
        return sorted(stems, key=float)

    def _frames(self, seq_name: str) -> list:
        """Frame list for a camera stream, listed lazily on first access and
        cached in ``data_store``: [(rgb_path, depth_path, cam_path, timestamp)]
        in temporal order. ``timestamp`` is the float-second filename stem
        (= frame_index / 30).

        Raises ValueError if the stream has fewer than ``min_num_images``
        frames (construction defers per-stream listing, so the eager
        TUM/7-Scenes-style skip-at-init is not possible here; the export is
        uniformly 300 frames/stream, so this should never trigger).
        """
        frames = self.data_store[seq_name]
        if frames is None:
            stream_dir = os.path.join(self._split_dir, seq_name)
            stems = [
                os.path.basename(p)[: -len(".png")]
                for p in glob.glob(os.path.join(stream_dir, "rgb", "*.png"))
            ]
            frames = [
                (
                    os.path.join(stream_dir, "rgb", stem + ".png"),
                    os.path.join(stream_dir, "depth", stem + ".npy"),
                    os.path.join(stream_dir, "cam", stem + ".npz"),
                    float(stem),
                )
                for stem in self.sort_frame_stems(stems)
            ]
            if len(frames) < self.min_num_images:
                raise ValueError(
                    f"Dynamic Replica stream {seq_name!r}: only {len(frames)} frames "
                    f"(< {self.min_num_images}) under {stream_dir}"
                )
            self.data_store[seq_name] = frames
        return frames

    def __init__(
        self,
        common_conf,
        split: str = "train",
        DYNAMIC_REPLICA_DIR: str = None,
        sequences=None,
        len_train: int = 100000,
        len_test: int = 10000,
        expand_ratio: int = 8,
        cameras=("left", "right"),
        intrinsics=None,
        min_num_images: int = 24,
    ):
        super().__init__(common_conf=common_conf)
        # per-dataset flags BaseDataset.process_one_image / get_data rely on
        self.training = common_conf.training
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.get_nearby = common_conf.get_nearby

        if DYNAMIC_REPLICA_DIR is None:
            raise ValueError("DYNAMIC_REPLICA_DIR must be specified")
        cameras = list(cameras)
        unknown = [c for c in cameras if c not in self._CAMERA_IDS]
        if unknown:
            raise ValueError(
                f"cameras must be a subset of {sorted(self._CAMERA_IDS)}, got {unknown}"
            )

        self.DYNAMIC_REPLICA_DIR = DYNAMIC_REPLICA_DIR
        # Only a 'train/' split exists on disk; `split` selects the virtual
        # epoch length, not the data (see module docstring).
        self._split_dir = os.path.join(DYNAMIC_REPLICA_DIR, "train")
        if not os.path.isdir(self._split_dir):
            raise ValueError(
                f"Dynamic Replica: expected a 'train/' split dir under "
                f"{DYNAMIC_REPLICA_DIR} (only on-disk split); not found"
            )
        self.expand_ratio = expand_ratio
        self.cameras = cameras
        self.intrinsics_override = intrinsics
        self.min_num_images = min_num_images
        self.len_train = len_train if split == "train" else len_test
        self.available_modalities = self.AVAILABLE

        # SCALABILITY: one directory listing enumerates all sequence names
        # (entry.is_dir() uses the dirent type -- no extra stat per entry).
        # Per-stream frame lists are deferred to first access (see _frames).
        patterns = sequences or ["*"]
        with os.scandir(self._split_dir) as it:
            seq_dirs = sorted(e.name for e in it if e.is_dir(follow_symlinks=False))

        self.data_store = {}
        for name in seq_dirs:
            for camera in self.cameras:
                stream = f"{name}/{camera}"  # key doubles as the inference output-dir name
                if any(
                    fnmatch.fnmatchcase(name, pat) or fnmatch.fnmatchcase(stream, pat)
                    for pat in patterns
                ):
                    self.data_store[stream] = None  # frames listed lazily

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)
        if self.sequence_list_len == 0:
            raise ValueError(
                f"No usable Dynamic Replica streams under {self._split_dir} "
                f"(sequences={patterns}, cameras={cameras})"
            )
        self._native_size_cache = {}

    def sequence_num_frames(self, local_idx: int) -> int:
        """Number of frames in the camera stream at ``local_idx`` of this
        vendor's ``sequence_list`` (used by ComposedDataset enumeration).
        Triggers the lazy frame listing for that one stream."""
        return len(self._frames(self.sequence_list[local_idx]))

    def native_image_size(self, local_idx: int = 0):
        """Native ``(H, W)`` of the source RGB frames for the stream at
        ``local_idx``, read lazily from the first frame's header and cached."""
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
        cam_id = self.camera_id_for_stream(seq_name)
        K_override = (
            self.dynamic_replica_intrinsics(self.intrinsics_override)
            if self.intrinsics_override is not None
            else None
        )

        images, depths, extrinsics, intrinsics = [], [], [], []
        cam_points, world_points, point_masks = [], [], []
        sky_masks, timestamps, camera_ids, original_sizes = [], [], [], []

        for i in ids:
            rgb_path, depth_path, cam_path, ts = frames[int(i)]
            image = read_image_cv2(rgb_path)  # cv2 drops the constant-255 alpha
            if image is None:
                # Globbed from the stream dir, so the file should exist; fail
                # loudly (a silent skip would yield fewer than img_per_seq frames
                # and break fixed-V batch stacking).
                raise FileNotFoundError(
                    f"Dynamic Replica: could not read image {rgb_path}"
                )
            depth_map = self.read_dynamic_replica_depth(depth_path)
            pose_w2c, K_frame = self.read_dynamic_replica_camera(cam_path)
            K = K_override if K_override is not None else K_frame
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
            sky_masks.append(depth_map < 0)  # indoor synthetic: always all-False (sky convention = depth<0)
            timestamps.append(ts)
            camera_ids.append(cam_id)
            original_sizes.append(original_size)

        return {
            "seq_name": "dynamic_replica_" + seq_name,
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
            "camera_ids": np.array(camera_ids, dtype=np.int32),
            "original_sizes": original_sizes,
            "is_metric": True,
            "is_video": True,
            "modalities": set(self.available_modalities),
        }
