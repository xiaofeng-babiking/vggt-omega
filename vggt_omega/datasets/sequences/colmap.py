"""COLMAP sparse-reconstruction vendor, implemented against BaseSequence.

A COLMAP model (``<seq>/sparse/0/{cameras,images,points3D}.bin``) is a
Structure-from-Motion output: registered images with absolute world->camera
poses, shared per-camera pinhole-ish intrinsics, and a sparse 3D point cloud.
This vendor exposes one *sensor per COLMAP camera_id* (the reference mip-NeRF-360
captures are single-camera => one sensor); frames are that camera's registered
images sorted by filename, addressed by 0-based position. Construction reads the
model once and touches no pixels; ``get_rgb`` decodes lazily.

pycolmap is imported lazily inside the adapter methods, so importing this module
never requires it (only constructing/using a ColmapSequence does).
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import networkx as nx
from PIL import Image
from scipy.spatial.transform import Rotation

from vggt_omega.datasets.sequences.base_sequence import BaseSequence, Modality
from vggt_omega.datasets.utils.se3_trajectory import NumpySE3Trajectory


class ColmapSequence(BaseSequence):
    """One COLMAP sparse reconstruction as a :class:`BaseSequence`.

    Sensor == COLMAP ``camera_id``; frames == that camera's registered images
    sorted by filename, addressed by 0-based position. Advertises
    ``RGB, POSE, INTRINSIC, EXTRINSIC, POINTCLOUD``. Depth / masks / tracks are
    absent in a sparse model and raise ``NotImplementedError``.
    """

    def __init__(
        self,
        data_root: str,
        seq_id: str,
        cache_dir: Optional[str] = None,
        sparse_dir: str = "sparse/0",
        images_dir: str = "images",
    ):
        """Open a COLMAP sequence.

        Args:
            data_root: directory containing the sequences.
            seq_id: sequence sub-directory (e.g. ``"garden"``).
            cache_dir: cache directory (unused; kept for the base signature).
            sparse_dir: model sub-dir holding ``{cameras,images,points3D}.bin``.
            images_dir: image sub-dir RGB is decoded from (``images_2/4/8`` supported;
                intrinsics are auto-rescaled to its on-disk resolution).
        """
        self._sparse_dir = sparse_dir
        self._images_dir = images_dir
        # Reconstruction is cached only across the three construction hooks, then
        # dropped so the instance stays lightweight AND picklable (DataLoader
        # workers pickle the dataset; a live pycolmap object would break that).
        self._reconstruction = None
        self._images_by_cam: Dict[int, list] = {}
        super().__init__(data_root, seq_id, cache_dir)
        # Drop the pycolmap objects used only during the construction hooks so the
        # instance stays lightweight and picklable (DataLoader workers pickle it).
        self._reconstruction = None
        self._images_by_cam = {}

    # -- pycolmap adapter (the ONLY place pycolmap is touched) --------------- #
    def _open_reconstruction(self):
        """Load the model dir into a ``pycolmap.Reconstruction`` (lazy import)."""
        import pycolmap  # heavy + optional: import only when actually opening

        model_dir = os.path.join(self._seq_dir, self._sparse_dir)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"COLMAP model dir not found: {model_dir}")
        return pycolmap.Reconstruction(model_dir)

    @staticmethod
    def _camera_intrinsic(camera) -> np.ndarray:
        """``[fx, fy, cx, cy]`` for any camera model via the 3x3 calibration matrix."""
        K = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)

    @staticmethod
    def _image_w2c(image) -> Tuple[np.ndarray, np.ndarray]:
        """``(R_w2c (3,3), t_w2c (3,))`` from a pycolmap ``Image``.

        Uses ``image.cam_from_world`` -> ``Rigid3d`` and its ``.matrix()`` (3x4
        ``[R|t]``, world->cam, convention-unambiguous). ``cam_from_world`` is a
        *method* in pycolmap >=4.1 (verified 4.1.0) and a ``Rigid3d`` property in
        <=4.0.4, so we call it when callable. Legacy ``image.qvec``/``image.tvec``
        (removed in >=4.1) are the last-resort fallback. The downstream
        ``NumpySE3Trajectory.inverse`` conversion stays unchanged.
        """
        cfw = getattr(image, "cam_from_world", None)
        if cfw is not None:
            rigid = cfw() if callable(cfw) else cfw  # >=4.1: method; <=4.0.4: property
            M = np.asarray(rigid.matrix(), dtype=np.float64)  # (3, 4) [R | t]
            return M[:3, :3], M[:3, 3]
        q_wxyz = np.asarray(image.qvec, dtype=np.float64)
        R = Rotation.from_quat(q_wxyz[[1, 2, 3, 0]]).as_matrix()  # wxyz -> xyzw
        return R, np.asarray(image.tvec, dtype=np.float64).reshape(3)

    # -- load hooks ---------------------------------------------------------- #
    def load_manifest(self) -> Dict[Union[int, str], Dict[Modality, list]]:
        """Group registered images by ``camera_id``, filename-sorted; cache the
        reconstruction + per-camera image lists for the other two hooks."""
        self._reconstruction = self._open_reconstruction()
        if len(self._reconstruction.images) == 0:
            raise ValueError(
                f"COLMAP reconstruction has no registered images: {self._seq_dir}"
            )
        by_cam: Dict[int, list] = {}
        for image in self._reconstruction.images.values():
            by_cam.setdefault(int(image.camera_id), []).append(image)

        manifest: Dict[Union[int, str], Dict[Modality, list]] = {}
        for cam_id, imgs in by_cam.items():
            imgs = sorted(imgs, key=lambda im: im.name)
            self._images_by_cam[cam_id] = imgs
            manifest[cam_id] = {
                Modality.NAME: list(range(len(imgs))),  # positional ids (see contract)
                Modality.RGB: [
                    os.path.join(self._seq_dir, self._images_dir, im.name) for im in imgs
                ],
            }
        return manifest

    def load_calibration_tree(self) -> nx.DiGraph:
        """One node per ``camera_id`` carrying ``intrinsic = [fx, fy, cx, cy]``,
        rescaled to the on-disk ``images_dir`` resolution. No edges (COLMAP images
        carry absolute poses; cameras are independent)."""
        tree = nx.DiGraph()
        for cam_id in self._manifest:
            camera = self._reconstruction.cameras[cam_id]
            intrinsic = self._camera_intrinsic(camera)
            h, w = self.get_rgb_size(cam_id)  # base reads first RGB header (imagesize)
            cw, ch = int(camera.width), int(camera.height)
            if (w, h) != (cw, ch):
                sx, sy = w / float(cw), h / float(ch)
                if abs(sx - sy) > 1e-2 * max(sx, sy):
                    raise ValueError(
                        f"images_dir {self._images_dir!r} size {(w, h)} is not a uniform "
                        f"rescale of COLMAP camera {(cw, ch)} for camera {cam_id}"
                    )
                intrinsic = self.rescale_intrinsic(intrinsic, (cw, ch), (w, h))
            tree.add_node(cam_id, intrinsic=intrinsic.astype(np.float32))
        return tree

    def load_sensor_poses(self) -> Dict[Union[int, str], NumpySE3Trajectory]:
        """Per-camera camera-to-world trajectory (index-parameterised, no timestamps).

        COLMAP stores world->camera poses; the base wants camera-to-world. We build
        the world->camera trajectory (batched matrix->quaternion) and let
        :meth:`NumpySE3Trajectory.inverse` do the SE(3) inversion in one vectorised
        pass — it also swaps the ``world``/``cam`` frame labels, so the returned
        trajectory is self-consistently tagged ``source=cam_id, target="world"``.
        """
        poses: Dict[Union[int, str], NumpySE3Trajectory] = {}
        for cam_id, imgs in self._images_by_cam.items():
            mats = [self._image_w2c(im) for im in imgs]
            R_w2c = np.stack([R for R, _ in mats], axis=0)  # (N, 3, 3)
            t_w2c = np.stack([t for _, t in mats], axis=0)  # (N, 3)
            quat_xyzw = Rotation.from_matrix(R_w2c).as_quat()  # scalar-last, batched
            w2c_rows = np.concatenate([quat_xyzw, t_w2c], axis=1)  # (N, 7)
            w2c = NumpySE3Trajectory(
                timestamps=None, poses=w2c_rows, source="world", target=cam_id, normalize=True
            )
            poses[cam_id] = w2c.inverse()
        return poses

    # -- discovery ----------------------------------------------------------- #
    def get_sensors(self) -> List[Union[int, str]]:
        return sorted(self._manifest.keys())

    def get_modalities(self, sensor_id: Union[int, str]) -> Set[Modality]:
        return {
            Modality.RGB,
            Modality.POSE,
            Modality.INTRINSIC,
            Modality.EXTRINSIC,
            Modality.POINTCLOUD,
        }

    def get_length(self, sensor_id: Union[int, str]) -> int:
        return len(self._manifest[sensor_id][Modality.NAME])

    # -- per-frame getters --------------------------------------------------- #
    def get_rgb(self, sensor_id: Union[int, str], frame_index: int) -> np.ndarray:
        """Decode the color frame -> ``(H, W, 3)`` uint8, by position."""
        rgb_file = self._manifest[sensor_id][Modality.RGB][frame_index]
        with Image.open(rgb_file) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def get_pose(
        self, sensor_id: Union[int, str], frame_index: int
    ) -> NumpySE3Trajectory:
        """Single-frame camera-to-world :class:`NumpySE3Trajectory` slice, by position."""
        return self._sensor_poses[sensor_id][frame_index]

    def get_poses(self, sensor_id: Union[int, str]) -> NumpySE3Trajectory:
        """Full camera-to-world trajectory for a sensor."""
        return self._sensor_poses[sensor_id]

    # -- per-sequence products ----------------------------------------------- #
    def get_pointcloud(self, sensor_id: Union[int, str]) -> np.ndarray:
        """Full sparse cloud ``(N, 3)`` float32 (shared across sensors).

        Reopens the reconstruction (not kept on ``self`` — see ``__init__``); this
        getter is off the hot path and consumed by nothing today.
        """
        rec = self._open_reconstruction()
        pts = [np.asarray(p.xyz, dtype=np.float64) for p in rec.points3D.values()]
        if not pts:
            return np.empty((0, 3), dtype=np.float32)
        return np.stack(pts, axis=0).astype(np.float32).reshape(-1, 3)

    # -- not provided by a sparse COLMAP model ------------------------------- #
    def get_rgb_semantic_mask(self, sensor_id, frame_index):
        raise NotImplementedError("COLMAP provides no semantic masks")

    def get_rgb_dynamic_mask(self, sensor_id, frame_index):
        raise NotImplementedError("COLMAP provides no dynamic masks")

    def get_rgb_valid_mask(self, sensor_id, frame_index):
        raise NotImplementedError("COLMAP provides no valid masks")

    def get_depth(self, sensor_id, frame_index):
        raise NotImplementedError("sparse COLMAP provides no depth")

    def get_depth_confidence(self, sensor_id, frame_index):
        raise NotImplementedError("sparse COLMAP provides no depth confidence")

    def get_tracks(self, sensor_id):
        raise NotImplementedError(
            "ColmapSequence does not expose 2D tracks (see design doc: not consumed, "
            "dense track matrix impractical)"
        )

    # -- discovery classmethod ----------------------------------------------- #
    @classmethod
    def discover(cls, data_root: str, patterns: Optional[List[str]] = None) -> List[str]:
        """Sequence ids = sub-dirs holding a COLMAP model (``sparse/0/cameras.bin``
        or ``sparse/cameras.bin``), filtered by the glob ``patterns`` (``["*"]``)."""
        names = set()
        for pat in patterns or ["*"]:
            for d in glob.glob(os.path.join(data_root, pat)):
                if not os.path.isdir(d):
                    continue
                if os.path.isfile(os.path.join(d, "sparse", "0", "cameras.bin")) or \
                   os.path.isfile(os.path.join(d, "sparse", "cameras.bin")):
                    names.add(os.path.relpath(d, data_root))
        return sorted(names)

    def __repr__(self) -> str:
        return (
            f"ColmapSequence(seq_id={self._seq_id!r}, "
            f"cameras={len(self._manifest)}, "
            f"frames={sum(len(v[Modality.NAME]) for v in self._manifest.values())})"
        )
