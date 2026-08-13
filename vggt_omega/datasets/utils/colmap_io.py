"""Self-contained COLMAP writers: sparse model (binary) + native SQLite database.

Ported from COLMAP itself (``awesome-3d-vision/colmap`` @ ``musa``) so vggt-omega can
emit COLMAP artifacts with **no colmap / pycolmap dependency** -- the formats are
borrowed, not linked. Everything here is plain ``struct`` / ``sqlite3`` / numpy.

Provenance (read from the COLMAP tree; keep in sync if the upstream format moves):

- ``src/colmap/scene/reconstruction_io_binary.cc`` -- ``Write{Cameras,Images,Points3D}Binary``
  give the exact little-endian byte layout replicated in :func:`write_model_binary`.
- ``src/colmap/scene/database_sqlite.cc`` -- the ``CREATE TABLE`` DDL replicated verbatim
  in :attr:`COLMAPDatabase.SCHEMA`.
- ``src/colmap/mvs/mat.cc`` + ``mat.h`` -- ``Mat<float>::Write/Read``: the dense-workspace
  depth/normal map layout replicated in :func:`write_colmap_mat` / :func:`read_colmap_mat`
  (ASCII ``w&h&c&`` header, then float32 LE channel planes).
- ``src/colmap/util/types.h`` -- id widths: ``camera_t``/``image_t``/``frame_t``/
  ``point2D_t`` are ``uint32``, ``point3D_t`` is ``uint64``, ``kMaxNumImages`` = 2**31-1.
- ``src/colmap/sensor/models.h`` -- ``PINHOLE`` model id = 1, params ``[fx, fy, cx, cy]``.
- ``src/colmap/geometry/pose_prior.h`` -- ``CoordinateSystem``: UNDEFINED=-1, WGS84=0,
  CARTESIAN=1.

Model layout: we write the classic three files (``cameras.bin`` / ``images.bin`` /
``points3D.bin``). Current COLMAP additionally writes ``rigs.bin`` / ``frames.bin``, but
its reader guards both with ``ExistsFile`` (``Reconstruction::ReadBinary``) and
synthesizes a trivial rig for legacy models -- verified against pycolmap 4.1.0, which
loads a rigs/frames-less model reporting ``num_rigs = 1``. Three files therefore stay
maximally compatible (and are exactly what ``ColmapSequence`` reads back).

Pose convention: COLMAP stores **world-to-camera** ``[R|t]`` as a quaternion
``(qw, qx, qy, qz)`` plus translation -- the same convention vggt-omega's
``pred_extrinsics`` already use, so no inversion happens here.
"""

from __future__ import annotations

import os
import sqlite3
import struct
from typing import Dict, List, NamedTuple, Optional, Sequence

import cv2
import numpy as np

from vggt_omega.utils.geometry import world_to_camera_to_camera_to_world

# --- COLMAP constants (borrowed; see module docstring for provenance) --------- #
CAMERA_MODEL_PINHOLE = 1  # sensor/models.h: PINHOLE, params [fx, fy, cx, cy]
SENSOR_TYPE_CAMERA = 0  # sensor/sensor.h: SensorType::CAMERA
POSE_PRIOR_CARTESIAN = 1  # geometry/pose_prior.h: CoordinateSystem::CARTESIAN
MAX_NUM_IMAGES = 2**31 - 1  # util/types.h: kMaxNumImages = int32 max
INVALID_POINT3D_ID = 2**64 - 1  # util/types.h: kInvalidPoint3DId (uint64 max)


class ColmapCamera(NamedTuple):
    """One PINHOLE camera. ``params`` is ``[fx, fy, cx, cy]`` (pixels)."""

    camera_id: int
    width: int
    height: int
    params: np.ndarray
    model_id: int = CAMERA_MODEL_PINHOLE


class ColmapImage(NamedTuple):
    """One registered image. ``qvec`` is ``(qw, qx, qy, qz)`` and ``tvec`` ``(tx, ty, tz)``
    of the **world-to-camera** transform. ``xys`` is ``(N, 2)`` keypoints and
    ``point3D_ids`` the ``(N,)`` point3D id per keypoint (``-1`` -> no 3D point)."""

    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


class ColmapPoint3D(NamedTuple):
    """One 3D point. ``track`` is a list of ``(image_id, point2D_idx)``."""

    point3D_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    track: Sequence


# --- geometry ---------------------------------------------------------------- #
def rotmat_to_qvec(rot: np.ndarray) -> np.ndarray:
    """``(3, 3)`` rotation matrix -> COLMAP quaternion ``(qw, qx, qy, qz)``.

    Shepperd's method (branch on the largest diagonal term) for numerical stability;
    the sign is normalized to ``qw >= 0`` so the quaternion is unique.
    """
    rot = np.asarray(rot, dtype=np.float64)
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw, qx = 0.25 / s, (rot[2, 1] - rot[1, 2]) * s
        qy, qz = (rot[0, 2] - rot[2, 0]) * s, (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw, qx = (rot[2, 1] - rot[1, 2]) / s, 0.25 * s
        qy, qz = (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw, qx = (rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s
        qy, qz = 0.25 * s, (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw, qx = (rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s
        qy, qz = (rot[1, 2] + rot[2, 1]) / s, 0.25 * s
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    if qvec[0] < 0:
        qvec = -qvec
    return qvec / np.linalg.norm(qvec)


# --- sparse model: binary writers -------------------------------------------- #
# Byte layout mirrors reconstruction_io_binary.cc Write*Binary, little-endian.
def _pack(fmt: str, *vals) -> bytes:
    return struct.pack("<" + fmt, *vals)


def write_cameras_binary(cameras: Sequence[ColmapCamera], path: str) -> None:
    """``cameras.bin``: uint64 count, then per camera
    ``camera_id:u32, model_id:i32, width:u64, height:u64, params:f64[]``."""
    with open(path, "wb") as f:
        f.write(_pack("Q", len(cameras)))
        for cam in sorted(cameras, key=lambda c: c.camera_id):
            f.write(_pack("iiQQ", cam.camera_id, cam.model_id, cam.width, cam.height))
            f.write(_pack(f"{len(cam.params)}d", *np.asarray(cam.params, np.float64)))


def write_images_binary(images: Sequence[ColmapImage], path: str) -> None:
    """``images.bin``: uint64 count, then per image ``image_id:u32, qvec:f64[4],
    tvec:f64[3], camera_id:u32, name:cstring, num_points2D:u64, (x:f64, y:f64,
    point3D_id:u64) * num_points2D``."""
    with open(path, "wb") as f:
        f.write(_pack("Q", len(images)))
        for img in sorted(images, key=lambda i: i.image_id):
            q = np.asarray(img.qvec, np.float64)
            t = np.asarray(img.tvec, np.float64)
            f.write(_pack("i", img.image_id))
            f.write(_pack("7d", q[0], q[1], q[2], q[3], t[0], t[1], t[2]))
            f.write(_pack("i", img.camera_id))
            f.write(img.name.encode("utf-8") + b"\x00")  # null-terminated
            xys = np.asarray(img.xys, np.float64).reshape(-1, 2)
            ids = np.asarray(img.point3D_ids).reshape(-1)
            f.write(_pack("Q", xys.shape[0]))
            for (x, y), pid in zip(xys, ids):
                # -1 (no 3D point) maps to COLMAP's uint64 kInvalidPoint3DId.
                pid = INVALID_POINT3D_ID if int(pid) < 0 else int(pid)
                f.write(_pack("ddQ", float(x), float(y), pid))


def write_points3d_binary(points3D: Sequence[ColmapPoint3D], path: str) -> None:
    """``points3D.bin``: uint64 count, then per point ``point3D_id:u64, xyz:f64[3],
    rgb:u8[3], error:f64, track_len:u64, (image_id:u32, point2D_idx:u32) * track_len``."""
    with open(path, "wb") as f:
        f.write(_pack("Q", len(points3D)))
        for p in sorted(points3D, key=lambda p: p.point3D_id):
            xyz = np.asarray(p.xyz, np.float64)
            rgb = np.asarray(p.rgb, np.uint8)
            f.write(_pack("Q", p.point3D_id))
            f.write(_pack("3d", xyz[0], xyz[1], xyz[2]))
            f.write(_pack("3B", int(rgb[0]), int(rgb[1]), int(rgb[2])))
            f.write(_pack("d", float(p.error)))
            f.write(_pack("Q", len(p.track)))
            for image_id, point2D_idx in p.track:
                f.write(_pack("ii", int(image_id), int(point2D_idx)))


def write_model_binary(
    model_dir: str,
    cameras: Sequence[ColmapCamera],
    images: Sequence[ColmapImage],
    points3D: Sequence[ColmapPoint3D],
) -> None:
    """Write a COLMAP sparse model (``cameras.bin`` / ``images.bin`` / ``points3D.bin``)."""
    os.makedirs(model_dir, exist_ok=True)
    write_cameras_binary(cameras, os.path.join(model_dir, "cameras.bin"))
    write_images_binary(images, os.path.join(model_dir, "images.bin"))
    write_points3d_binary(points3D, os.path.join(model_dir, "points3D.bin"))


# --- dense workspace: mvs Mat<float> depth / normal maps ---------------------- #
def write_colmap_mat(path: str, array: np.ndarray) -> None:
    """Write ``(H, W)`` or ``(H, W, C)`` float data as COLMAP's ``mvs::Mat<float>``
    file (``src/colmap/mvs/mat.cc``): the ASCII header ``"<width>&<height>&<channels>&"``
    followed by float32 little-endian data in **slice-major** order --
    ``Mat::Get(row, col, slice)`` indexes ``data[slice*W*H + row*W + col]``, so the
    payload is whole channel planes, not interleaved pixels. This is the format of
    ``stereo/depth_maps/*.bin`` (C=1) and ``stereo/normal_maps/*.bin`` (C=3) in a
    COLMAP dense workspace."""
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[..., None]
    assert array.ndim == 3, f"expected (H, W[, C]), got {array.shape}"
    height, width, channels = array.shape
    planes = np.ascontiguousarray(np.moveaxis(array, -1, 0).astype("<f4"))
    with open(path, "wb") as f:
        f.write(f"{width}&{height}&{channels}&".encode("ascii"))
        f.write(planes.tobytes())


def read_colmap_mat(path: str) -> np.ndarray:
    """Read an ``mvs::Mat<float>`` file back -> ``(H, W, C)`` float32 (the inverse of
    :func:`write_colmap_mat`)."""
    with open(path, "rb") as f:
        blob = f.read()
    dims, start = [], 0
    for _ in range(3):
        end = blob.index(b"&", start)
        dims.append(int(blob[start:end]))
        start = end + 1
    width, height, channels = dims
    data = np.frombuffer(blob, dtype="<f4", offset=start, count=width * height * channels)
    return np.moveaxis(data.reshape(channels, height, width), 0, -1).copy()


# --- native SQLite database --------------------------------------------------- #
def _array_blob(array: Optional[np.ndarray]) -> Optional[bytes]:
    """COLMAP stores matrices as raw little-endian element bytes (row-major)."""
    if array is None:
        return None
    return np.ascontiguousarray(array).tobytes()


class COLMAPDatabase(sqlite3.Connection):
    """COLMAP's native SQLite database, schema borrowed verbatim from
    ``src/colmap/scene/database_sqlite.cc``.

    Open with :meth:`connect`, call :meth:`create_tables`, insert, then ``commit()``.
    Only the tables vggt-omega populates are exercised (cameras / images / keypoints /
    pose_priors); the rest are still created so the file is a structurally valid COLMAP
    database that its tooling will open.

    Note the schema's ``images`` table carries **no pose** (this COLMAP moved poses to
    rigs/frames), and ``pose_priors`` stores only a *position* -- so the database cannot
    represent a full rotation. The sparse model is the artifact that carries poses.
    """

    # Verbatim DDL (database_sqlite.cc). image_id_check interpolates kMaxNumImages.
    SCHEMA: Dict[str, str] = {
        "rigs": """CREATE TABLE IF NOT EXISTS rigs
   (rig_id               INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    ref_sensor_id        INTEGER                             NOT NULL,
    ref_sensor_type      INTEGER                             NOT NULL)""",
        "rig_sensors": """CREATE TABLE IF NOT EXISTS rig_sensors
   (rig_id               INTEGER                             NOT NULL,
    sensor_id            INTEGER                             NOT NULL,
    sensor_type          INTEGER                             NOT NULL,
    sensor_from_rig      BLOB,
    FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE)""",
        "cameras": """CREATE TABLE IF NOT EXISTS cameras
   (camera_id            INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    model                INTEGER                             NOT NULL,
    width                INTEGER                             NOT NULL,
    height               INTEGER                             NOT NULL,
    params               BLOB,
    prior_focal_length   INTEGER                             NOT NULL)""",
        "frames": """CREATE TABLE IF NOT EXISTS frames
   (frame_id             INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    rig_id               INTEGER                             NOT NULL,
    FOREIGN KEY(rig_id) REFERENCES rigs(rig_id) ON DELETE CASCADE)""",
        "frame_data": """CREATE TABLE IF NOT EXISTS frame_data
   (frame_id             INTEGER                             NOT NULL,
    data_id              INTEGER                             NOT NULL,
    sensor_id            INTEGER                             NOT NULL,
    sensor_type          INTEGER                             NOT NULL,
    FOREIGN KEY(frame_id) REFERENCES frames(frame_id) ON DELETE CASCADE)""",
        "images": f"""CREATE TABLE IF NOT EXISTS images
   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,
    name       TEXT                                NOT NULL UNIQUE,
    camera_id  INTEGER                             NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {MAX_NUM_IMAGES}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))""",
        "pose_priors": """CREATE TABLE IF NOT EXISTS pose_priors
   (pose_prior_id              INTEGER  PRIMARY KEY  NOT NULL,
    corr_data_id               INTEGER               NOT NULL,
    corr_sensor_id             INTEGER               NOT NULL,
    corr_sensor_type           INTEGER               NOT NULL,
    position                   BLOB,
    position_covariance        BLOB,
    gravity                    BLOB,
    coordinate_system          INTEGER               NOT NULL)""",
        "keypoints": """CREATE TABLE IF NOT EXISTS keypoints
   (image_id  INTEGER  PRIMARY KEY  NOT NULL,
    rows      INTEGER               NOT NULL,
    cols      INTEGER               NOT NULL,
    data      BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)""",
        "descriptors": """CREATE TABLE IF NOT EXISTS descriptors
   (image_id      INTEGER  PRIMARY KEY  NOT NULL,
    type          INTEGER               NOT NULL,
    rows          INTEGER               NOT NULL,
    cols          INTEGER               NOT NULL,
    data          BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)""",
        "matches": """CREATE TABLE IF NOT EXISTS matches
   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,
    rows     INTEGER               NOT NULL,
    cols     INTEGER               NOT NULL,
    data     BLOB)""",
        "two_view_geometries": """CREATE TABLE IF NOT EXISTS two_view_geometries
   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,
    rows     INTEGER               NOT NULL,
    cols     INTEGER               NOT NULL,
    data     BLOB,
    config   INTEGER               NOT NULL,
    F        BLOB,
    E        BLOB,
    H        BLOB,
    qvec     BLOB,
    tvec     BLOB)""",
    }

    @staticmethod
    def connect(path: str) -> "COLMAPDatabase":
        return sqlite3.connect(path, factory=COLMAPDatabase)

    def create_tables(self) -> None:
        for ddl in self.SCHEMA.values():
            self.executescript(ddl + ";")

    def add_camera(
        self,
        camera_id: int,
        model: int,
        width: int,
        height: int,
        params: np.ndarray,
        prior_focal_length: bool = True,
    ) -> int:
        self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                int(width),
                int(height),
                _array_blob(np.asarray(params, np.float64)),
                int(prior_focal_length),
            ),
        )
        return camera_id

    def add_image(self, image_id: int, name: str, camera_id: int) -> int:
        self.execute(
            "INSERT INTO images VALUES (?, ?, ?)", (image_id, name, camera_id)
        )
        return image_id

    def add_keypoints(self, image_id: int, keypoints: np.ndarray) -> None:
        """``keypoints`` is ``(N, 2|4|6)`` float32 (COLMAP stores x, y[, scale, orient...])."""
        kp = np.ascontiguousarray(np.asarray(keypoints, np.float32))
        assert kp.ndim == 2 and kp.shape[1] in (2, 4, 6), kp.shape
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id, kp.shape[0], kp.shape[1], _array_blob(kp)),
        )

    def add_rig(self, rig_id: int, ref_sensor_id: int) -> int:
        """A rig whose reference sensor is the camera ``ref_sensor_id``."""
        self.execute(
            "INSERT INTO rigs VALUES (?, ?, ?)",
            (rig_id, ref_sensor_id, SENSOR_TYPE_CAMERA),
        )
        return rig_id

    def add_frame(self, frame_id: int, rig_id: int) -> int:
        self.execute("INSERT INTO frames VALUES (?, ?)", (frame_id, rig_id))
        return frame_id

    def add_frame_data(self, frame_id: int, data_id: int, sensor_id: int) -> None:
        """Bind one sensor measurement (here: image ``data_id`` from camera
        ``sensor_id``) to ``frame_id``."""
        self.execute(
            "INSERT INTO frame_data VALUES (?, ?, ?, ?)",
            (frame_id, data_id, sensor_id, SENSOR_TYPE_CAMERA),
        )

    def add_pose_prior(
        self,
        frame_id: int,
        data_id: int,
        sensor_id: int,
        position: np.ndarray,
        coordinate_system: int = POSE_PRIOR_CARTESIAN,
    ) -> None:
        """Position-only prior for ``frame_id`` (the schema holds no rotation).

        Current COLMAP associates pose priors with **frames**, not images -- its reader
        rejects image-keyed priors ("pose priors are now associated with frames"), so
        ``pose_prior_id`` is the frame id and ``corr_data_id`` / ``corr_sensor_id``
        identify the camera measurement that frame carries.
        """
        self.execute(
            "INSERT INTO pose_priors VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                frame_id,
                data_id,
                sensor_id,
                SENSOR_TYPE_CAMERA,
                _array_blob(np.asarray(position, np.float64)),
                None,
                None,
                int(coordinate_system),
            ),
        )


def write_database(
    db_path: str,
    cameras: Sequence[ColmapCamera],
    images: Sequence[ColmapImage],
    positions: Optional[Dict[int, np.ndarray]] = None,
) -> None:
    """Write a native COLMAP SQLite database: ``cameras`` / ``images`` / ``keypoints``,
    plus the ``rigs`` / ``frames`` / ``frame_data`` scaffolding and (when ``positions``
    is given) frame-keyed ``pose_priors``.

    Each image becomes its own single-sensor rig + frame (``rig_id == frame_id ==
    image_id``), which is what a monocular capture is: one camera measurement per
    timestep. That scaffolding is required because current COLMAP keys pose priors by
    frame, not image. ``positions[image_id]`` is the camera centre in world coords.

    Reminder: this file holds **no rotations** -- the sparse model carries the poses.
    Overwrites ``db_path``.
    """
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = COLMAPDatabase.connect(db_path)
    try:
        db.create_tables()
        for cam in cameras:
            db.add_camera(cam.camera_id, cam.model_id, cam.width, cam.height, cam.params)
        for img in images:
            db.add_image(img.image_id, img.name, img.camera_id)
            xys = np.asarray(img.xys, np.float32).reshape(-1, 2)
            if xys.shape[0]:
                db.add_keypoints(img.image_id, xys)
            # One rig + one frame per image (monocular: one camera datum per timestep).
            db.add_rig(img.image_id, img.camera_id)
            db.add_frame(img.image_id, img.image_id)
            db.add_frame_data(img.image_id, img.image_id, img.camera_id)
            if positions is not None and img.image_id in positions:
                db.add_pose_prior(
                    frame_id=img.image_id,
                    data_id=img.image_id,
                    sensor_id=img.camera_id,
                    position=positions[img.image_id],
                )
        db.commit()
    finally:
        db.close()


# --- prediction -> COLMAP export --------------------------------------------
# High-level orchestration over the format writers above. Moved from the retired
# inference_common.py; the inference entrypoint calls dump_as_colmap, which builds
# the sparse model, database, and ready-to-fuse dense workspace via those writers.

def fake_depth_map_to_camera_normals(
    depth_maps: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Per-pixel surface normals from depth ``(S, H, W)`` -> ``(S, H, W, 3)`` float32,
    in each frame's OpenCV camera frame.

    Each pixel is lifted to the camera-frame point ``P = depth * ((x-cx)/fx,
    (y-cy)/fy, 1)`` -- the same integer-pixel ray convention as
    :func:`unproject_depth_map_to_point_map` -- and the normal is
    ``normalize(dP/dy x dP/dx)`` from central differences, oriented toward the camera
    (``n . P < 0``). That is COLMAP's normal-map convention: mvs normal maps live in
    the camera frame and ``fusion.cc`` rotates them to world with ``R^T``, so
    cross-view normal-consistency checks only hold if the per-frame normals are truly
    camera-frame (a constant map like ``(0,0,-1)`` would diverge between rotated
    views).

    Pixels with invalid depth (non-finite or <= 0) get a zero normal. Valid pixels
    whose neighbourhood cannot be differenced (rims of holes, isolated pixels --
    their central difference touches an invalid neighbour) fall back to the
    fronto-parallel ``(0, 0, -1)``, COLMAP's own patch-match initialization.
    """
    num_f, height, width = depth_maps.shape
    normals = np.zeros((num_f, height, width, 3), dtype=np.float32)
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    for i in range(num_f):
        depth = np.asarray(depth_maps[i], dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0)
        if not valid.any():
            continue
        d = np.where(valid, depth, np.nan)
        K = intrinsics[i]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        points = np.stack([(x - cx) / fx * d, (y - cy) / fy * d, d], axis=-1)
        if height >= 2 and width >= 2:
            d_dy, d_dx = np.gradient(points, axis=(0, 1))
            n = np.cross(d_dy, d_dx)  # fronto-parallel plane -> (0, 0, -1)
        else:
            n = np.full_like(points, np.nan)  # nothing to difference against
        norm = np.linalg.norm(n, axis=-1)
        good = valid & np.isfinite(norm) & (norm > 0)
        n = np.where(good[..., None], n / np.where(good, norm, 1.0)[..., None], 0.0)
        # Orient toward the camera: flip wherever n . P > 0.
        dots = np.einsum("hwc,hwc->hw", n, np.nan_to_num(points))
        n = np.where((dots > 0)[..., None], -n, n)
        n[valid & ~good] = (0.0, 0.0, -1.0)
        normals[i] = n.astype(np.float32)
    return normals


def fake_extend_dense_tracks(
    world_points: np.ndarray,
    masked_depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    keep_masks: np.ndarray,
    pred_conf: np.ndarray,
    tracks_per_frame: int,
    max_depth_error: float,
    max_track_length: int,
):
    """Depth-validated cross-view tracks for the dense model's point subsample.

    ``stereo_fusion`` builds its image-overlap graph ONLY from shared sparse-point
    tracks (``mvs/model.cc ComputeSharedPoints``); with the single-image tracks the
    plain sparse export carries, every image has zero overlap, each fused point
    gathers one pixel, and the default ``min_num_pixels`` filter drops the entire
    cloud. So the dense model re-tracks a bounded subsample: per frame the
    ``tracks_per_frame`` highest-confidence surviving pixels, each projected into
    every other frame and kept where it lands in bounds on a pixel whose *written*
    depth agrees within ``max_depth_error`` (relative, like COLMAP's fusion check).

    When a point survives in more than ``max_track_length`` views, the kept ones are
    the views whose CAMERA CENTRES ARE NEAREST the source frame's -- not an even
    spread over the sequence. This matters more than it looks: fusion traverses only
    ``GetMaxOverlappingImages``, which ranks images by shared-track count alone, so
    whichever views a track keeps *become* that image's traversal targets. Spreading
    them evenly makes far-apart viewpoints an image's strongest "overlaps" and drops
    its immediate neighbours out of the list entirely; fusion then tries to gather
    support exactly where predicted depth agrees worst, every point falls under
    ``min_num_pixels``, and the cloud collapses. (Measured on garden: an even spread
    put neighbours in the top-50 only 0.5% of the time and left 80% of fused points
    with support 1.) Ranking by centre distance keeps the small-baseline neighbours
    fusion actually needs and makes no assumption about frame ordering.

    Cross-view observations are stored as the exact float projection of the point,
    so every track element reprojects onto its own observation by construction.

    Returns ``(own_pix, offsets, tracks, extra_xys, extra_pids)``: per-frame source
    pixel indices, per-frame point-id offsets, per-point track lists
    ``[(image_id, point2D_idx), ...]`` (source observation first), and per-image
    appended cross-view observations (coords / 1-based point ids).
    """
    num_f, height, width = masked_depth.shape
    own_pix, offsets = [], [0]
    for i in range(num_f):
        idx = np.flatnonzero(keep_masks[i].reshape(-1))
        if tracks_per_frame <= 0:
            idx = idx[:0]
        elif idx.size > tracks_per_frame:
            conf_i = pred_conf[i].reshape(-1)[idx]
            top = np.argpartition(-conf_i, tracks_per_frame - 1)[:tracks_per_frame]
            idx = np.sort(idx[top])
        own_pix.append(idx)
        offsets.append(offsets[-1] + idx.size)

    tracks = []
    for i in range(num_f):
        tracks.extend([(i + 1, k)] for k in range(own_pix[i].size))

    extra_xys = [[] for _ in range(num_f)]
    extra_pids = [[] for _ in range(num_f)]
    next_obs = [own_pix[j].size for j in range(num_f)]
    budget = max(0, max_track_length - 1)
    # Camera centres drive the "nearest views" choice below (C = -R^T t).
    centers = world_to_camera_to_camera_to_world(extrinsics)[:, :3, 3]

    for i in range(num_f):
        if own_pix[i].size == 0 or budget == 0:
            continue
        X = world_points[i].reshape(-1, 3)[own_pix[i]]
        js, src, uvs = [], [], []
        for j in range(num_f):
            if j == i:
                continue
            rot, trans = extrinsics[j][:3, :3], extrinsics[j][:3, 3]
            Xc = X @ rot.T + trans
            z = Xc[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = intrinsics[j][0, 0] * Xc[:, 0] / z + intrinsics[j][0, 2]
                v = intrinsics[j][1, 1] * Xc[:, 1] / z + intrinsics[j][1, 2]
            cand = np.flatnonzero((z > 0) & np.isfinite(u) & np.isfinite(v))
            if cand.size == 0:
                continue
            ui = np.rint(u[cand]).astype(np.int64)
            vi = np.rint(v[cand]).astype(np.int64)
            inb = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            cand, ui, vi = cand[inb], ui[inb], vi[inb]
            if cand.size == 0:
                continue
            depth_j = masked_depth[j][vi, ui].astype(np.float64)
            ok = (depth_j > 0) & (
                np.abs(z[cand] - depth_j) <= max_depth_error * depth_j
            )
            cand = cand[ok]
            if cand.size == 0:
                continue
            js.append(np.full(cand.size, j, dtype=np.int64))
            src.append(cand)
            uvs.append(np.stack([u[cand], v[cand]], axis=1))
        if not js:
            continue
        js, src, uvs = np.concatenate(js), np.concatenate(src), np.concatenate(uvs)

        # Per point: keep the `budget` validated views whose cameras sit closest to
        # frame i -- these become i's strongest overlaps, so they must be the views
        # fusion can actually agree with (see the docstring). Sort every entry by
        # (point, distance rank) and take each point's first `budget`.
        view_rank = np.empty(num_f, dtype=np.int64)
        view_rank[np.argsort(np.linalg.norm(centers - centers[i], axis=1), kind="stable")] = (
            np.arange(num_f)
        )
        order = np.argsort(src * num_f + view_rank[js], kind="stable")
        within = np.arange(js.size) - np.searchsorted(src[order], src[order], side="left")
        kept = order[within < budget]
        for j in np.unique(js[kept]):
            rows = kept[js[kept] == j]
            obs_idx = next_obs[j] + np.arange(rows.size)
            next_obs[j] += rows.size
            extra_xys[j].append(uvs[rows])
            gids = offsets[i] + src[rows]
            extra_pids[j].append(gids + 1)  # 1-based point3D ids
            for g, oi in zip(gids, obs_idx):
                tracks[g].append((j + 1, int(oi)))

    return own_pix, offsets, tracks, extra_xys, extra_pids


def dump_as_colmap(
    output_dir: str,
    images_hwc: np.ndarray,
    pred_depth_2d: np.ndarray,
    pred_conf: np.ndarray,
    pred_extrinsics: np.ndarray,
    pred_intrinsics: np.ndarray,
    world_points: np.ndarray,
    conf_percentile: float = 20.0,
    max_points: int = 2_000_000,
    image_names: Optional[List[str]] = None,
    dense_tracks_per_frame: int = 512,
    dense_track_max_depth_error: float = 0.01,
    dense_track_max_length: int = 16,
) -> dict:
    """Dump one sequence's predictions as COLMAP artifacts -> ``<output_dir>/colmap``.

    Writes a sparse model, the native SQLite database, and a ready-to-fuse dense
    workspace, via the self-contained writers in
    :mod:`vggt_omega.datasets.utils.colmap_io` (COLMAP's formats are *borrowed*, not
    depended on -- no colmap/pycolmap import)::

        <output_dir>/colmap/sparse/0/{cameras,images,points3D}.bin
        <output_dir>/colmap/database.db
        <output_dir>/colmap/dense/
            images/<name>                                # predicted RGB frames
            sparse/{cameras,images,points3D}.bin         # model with cross-view tracks
            stereo/depth_maps/<name>.geometric.bin       # conf-masked predicted depth
            stereo/normal_maps/<name>.geometric.bin      # camera-frame normals from depth
            stereo/consistency_graphs/                   # (empty; not needed by fusion)
            stereo/{fusion,patch-match}.cfg

    The dense workspace is exactly what ``colmap stereo_fusion --workspace_path
    <output_dir>/colmap/dense --input_type geometric`` consumes, so the predicted
    depth maps can be fused by COLMAP directly (no patch-match run needed). Its
    ``sparse`` model carries a per-frame top-confidence point subsample
    (``dense_tracks_per_frame``) whose tracks are extended by depth-validated
    reprojection (:func:`fake_extend_dense_tracks`) -- fusion derives its image-overlap
    graph purely from shared tracks, so without them it would fuse nothing. Depth
    maps zero out the pixels culled by ``conf_percentile`` (fusion treats
    ``depth <= 0`` as invalid); normal maps come from
    :func:`fake_depth_map_to_camera_normals` and are zeroed on the same mask.

    Two things to know before fusing (both measured on mip-NeRF-360 garden, 185
    frames):

    - **``conf_percentile`` deletes the far background.** Predicted confidence falls
      off with distance (``corr(depth, conf) = -0.65``), so the default 20% cutoff is
      not a uniform thinning -- it removes the *distant* pixels specifically: 88.5% of
      the farthest 10% are culled vs 3.4% of the nearest half, capping garden's depth
      maps at 2.20 m when the prediction runs to 8.57 m. The background architecture
      is simply absent from the fused cloud. Pass ``--conf_percentile 0`` to hand
      fusion the full depth map and let ITS multi-view checks do the filtering (that
      is what they are for); garden then fuses 495,787 points instead of 396,725,
      with the background intact.
    - **Expect to lower ``--StereoFusion.min_num_pixels``.** It defaults to 5, i.e. a
      point is kept only if 5 pixels agree on it, but VGGT's per-frame depth is only
      multi-view consistent to COLMAP's 1% tolerance for ~2.4 views per point on
      average -- so the default discards ~91% of candidates. Fusing garden with
      ``min_num_pixels=1`` yields 5.6M points vs 396k. (Relaxing the *consistency*
      knobs instead is counter-productive: looser normal/depth tolerances make each
      surviving point absorb more pixels, so the count goes DOWN.)

    Mapping from the VGGT-Omega prediction arrays (all frame-aligned, ``S`` frames):

    - **cameras**: one PINHOLE camera per frame (``camera_id == image_id``), params
      ``[fx, fy, cx, cy]`` straight from ``pred_intrinsics``. Per-frame cameras (rather
      than one shared camera) keep the predicted focal, which genuinely varies frame to
      frame (a broadcast camera zooms; the principal point is pinned to the image centre).
    - **images**: ``pred_extrinsics`` are already world-to-camera OpenCV ``[R|t]`` -- the
      exact COLMAP convention -- so the rotation is converted to a quaternion with no
      inversion. ``name`` is ``image_names[i]`` (default ``frame_XXXX.png``, matching the
      ``rgb/`` dump of ``--dump_per_frame``).
    - **points3D**: the confidence-ranked subsample described below, coloured from
      ``images_hwc`` and each carrying a single-image track back to its source pixel.
      Each image's ``xys`` / ``point3D_ids`` list only its own contributed points, so the
      tracks and the images' 2D observations are mutually consistent.

    Point selection: drop non-finite / non-positive-depth pixels, then the lowest
    ``conf_percentile`` percent by predicted confidence **per frame** (matching the
    per-frame PLY dump), then keep the globally highest-confidence ``max_points`` so the
    model stays loadable in COLMAP's tooling. ``max_points <= 0`` keeps everything;
    ``conf_percentile <= 0`` skips the per-frame cutoff.

    Returns a small stats dict (``num_images`` / ``num_points3D`` /
    ``num_dense_points3D`` / paths).
    """
    num_f, height, width = pred_depth_2d.shape
    colmap_dir = os.path.join(output_dir, "colmap")
    model_dir = os.path.join(colmap_dir, "sparse", "0")
    if image_names is None:
        image_names = [f"frame_{i:04d}.png" for i in range(num_f)]

    # --- select points: per-frame conf cutoff, then a global top-conf budget ------
    # Collect (frame, pixel) survivors first; ids are assigned after the global cap so
    # point3D ids stay contiguous. keep_stack is reused by the dense workspace below
    # (its depth maps zero exactly the pixels the point selection drops).
    keep_stack = np.zeros((num_f, height, width), dtype=bool)
    sel_frame, sel_pix, sel_conf = [], [], []
    for i in range(num_f):
        depth_i = pred_depth_2d[i].reshape(-1)
        conf_i = pred_conf[i].reshape(-1)
        pts_i = world_points[i].reshape(-1, 3)
        keep = np.isfinite(pts_i).all(axis=1) & np.isfinite(conf_i) & (depth_i > 0)
        if conf_percentile > 0 and keep.any():
            keep &= conf_i >= np.percentile(conf_i[keep], conf_percentile)
        keep_stack[i] = keep.reshape(height, width)
        idx = np.flatnonzero(keep)
        sel_frame.append(np.full(idx.shape, i, dtype=np.int64))
        sel_pix.append(idx)
        sel_conf.append(conf_i[idx])
    sel_frame = np.concatenate(sel_frame) if sel_frame else np.empty(0, np.int64)
    sel_pix = np.concatenate(sel_pix) if sel_pix else np.empty(0, np.int64)
    sel_conf = np.concatenate(sel_conf) if sel_conf else np.empty(0, np.float64)

    if max_points and sel_frame.size > max_points:
        top = np.argpartition(-sel_conf, max_points - 1)[:max_points]
        sel_frame, sel_pix = sel_frame[top], sel_pix[top]

    # --- build cameras / images / points3D ---------------------------------------
    cameras, images, points3D, positions = [], [], [], {}
    for i in range(num_f):
        image_id = i + 1  # COLMAP ids are 1-based
        K = pred_intrinsics[i]
        cameras.append(
            ColmapCamera(
                camera_id=image_id,
                width=int(width),
                height=int(height),
                params=np.array(
                    [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64
                ),
            )
        )
        # world-to-camera [R|t] -> COLMAP quaternion + translation (no inversion).
        rot, trans = pred_extrinsics[i][:3, :3], pred_extrinsics[i][:3, 3]

        mine = np.flatnonzero(sel_frame == i)
        pix = sel_pix[mine]
        ys, xs = np.divmod(pix, width)
        # Integer pixel coords, NOT the +0.5 pixel-centre convention: the 3D point was
        # unprojected as the ray through ((x - cx)/fx, (y - cy)/fy) (see
        # unproject_depth_map_to_point_map), so (x, y) is where it reprojects. Storing
        # x+0.5 would put every observation a systematic 0.707 px (sqrt(2)/2) off its own
        # point, which is exactly what a reprojection check catches -- self-consistency
        # with the geometry beats matching COLMAP's detector convention here.
        xys = np.stack([xs, ys], axis=1).astype(np.float64)
        # point3D ids are 1-based and assigned in this (frame-major) order.
        pids = (mine + 1).astype(np.int64)

        images.append(
            ColmapImage(
                image_id=image_id,
                qvec=rotmat_to_qvec(rot),
                tvec=np.asarray(trans, np.float64),
                camera_id=image_id,
                name=image_names[i],
                xys=xys,
                point3D_ids=pids,
            )
        )
        # camera centre in world coords: C = -R^T t (the only pose info a DB can hold).
        positions[image_id] = -rot.T @ np.asarray(trans, np.float64)

        rgb_flat = (images_hwc[i].reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
        xyz_flat = world_points[i].reshape(-1, 3)
        for local_idx, (p, pid) in enumerate(zip(pix, pids)):
            points3D.append(
                ColmapPoint3D(
                    point3D_id=int(pid),
                    xyz=xyz_flat[p],
                    rgb=rgb_flat[p],
                    error=-1.0,  # COLMAP's "unknown reprojection error" sentinel
                    track=[(image_id, local_idx)],
                )
            )

    write_model_binary(model_dir, cameras, images, points3D)
    db_path = os.path.join(colmap_dir, "database.db")
    write_database(db_path, cameras, images, positions=positions)

    # --- dense workspace: what `colmap stereo_fusion` consumes --------------------
    dense_dir = os.path.join(colmap_dir, "dense")
    stereo_dir = os.path.join(dense_dir, "stereo")
    dense_images_dir = os.path.join(dense_dir, "images")
    depth_maps_dir = os.path.join(stereo_dir, "depth_maps")
    normal_maps_dir = os.path.join(stereo_dir, "normal_maps")
    for d in (
        dense_images_dir,
        depth_maps_dir,
        normal_maps_dir,
        os.path.join(stereo_dir, "consistency_graphs"),
    ):
        os.makedirs(d, exist_ok=True)

    # Depth maps carry the prediction with culled pixels zeroed (fusion skips
    # depth <= 0). masked_depth is what cross-view track validation samples, so
    # tracks agree with the maps fusion will read.
    masked_depth = np.where(keep_stack, pred_depth_2d, 0.0).astype(np.float32)

    own_pix, offsets, tracks, extra_xys, extra_pids = fake_extend_dense_tracks(
        world_points,
        masked_depth,
        pred_extrinsics,
        pred_intrinsics,
        keep_stack,
        pred_conf,
        dense_tracks_per_frame,
        dense_track_max_depth_error,
        dense_track_max_length,
    )

    dense_images, dense_points3D = [], []
    for i in range(num_f):
        pix = own_pix[i]
        ys, xs = np.divmod(pix, width)
        # Own observations keep the integer-pixel convention of the sparse model
        # (see the xys note above); appended cross-view observations are the exact
        # float reprojections, so both kinds sit on their point by construction.
        own_xys = np.stack([xs, ys], axis=1).astype(np.float64).reshape(-1, 2)
        own_ids = offsets[i] + 1 + np.arange(pix.size, dtype=np.int64)
        xys = np.concatenate([own_xys] + [np.asarray(e) for e in extra_xys[i]])
        pids = np.concatenate([own_ids] + [np.asarray(p) for p in extra_pids[i]])
        dense_images.append(
            ColmapImage(
                image_id=i + 1,
                qvec=images[i].qvec,
                tvec=images[i].tvec,
                camera_id=i + 1,
                name=image_names[i],
                xys=xys,
                point3D_ids=pids,
            )
        )
        rgb_flat = (images_hwc[i].reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
        xyz_flat = world_points[i].reshape(-1, 3)
        for k, p in enumerate(pix):
            g = offsets[i] + k
            dense_points3D.append(
                ColmapPoint3D(
                    point3D_id=g + 1,
                    xyz=xyz_flat[p],
                    rgb=rgb_flat[p],
                    error=-1.0,
                    track=tracks[g],
                )
            )
    write_model_binary(os.path.join(dense_dir, "sparse"), cameras, dense_images, dense_points3D)

    for i in range(num_f):
        img_path = os.path.join(dense_images_dir, image_names[i])
        depth_path = os.path.join(depth_maps_dir, image_names[i] + ".geometric.bin")
        normal_path = os.path.join(normal_maps_dir, image_names[i] + ".geometric.bin")
        for p in (img_path, depth_path, normal_path):  # names may carry subdirs
            os.makedirs(os.path.dirname(p), exist_ok=True)
        bgr = cv2.cvtColor(
            (images_hwc[i] * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(img_path, bgr)
        write_colmap_mat(depth_path, masked_depth[i])
        # Normals come from the RAW depth (conf holes must not corrupt their
        # neighbours' gradients), masked like the depth map; per frame to avoid
        # materializing an (S, H, W, 3) stack.
        normals_i = fake_depth_map_to_camera_normals(
            pred_depth_2d[i : i + 1], pred_intrinsics[i : i + 1]
        )[0]
        normals_i[~keep_stack[i]] = 0.0
        write_colmap_mat(normal_path, normals_i)

    with open(os.path.join(stereo_dir, "fusion.cfg"), "w") as f:
        f.write("\n".join(image_names) + "\n")
    with open(os.path.join(stereo_dir, "patch-match.cfg"), "w") as f:
        for name in image_names:
            f.write(f"{name}\n__auto__, 20\n")

    return {
        "num_cameras": len(cameras),
        "num_images": len(images),
        "num_points3D": len(points3D),
        "num_dense_points3D": len(dense_points3D),
        "model_dir": model_dir,
        "database": db_path,
        "dense_dir": dense_dir,
    }


# --- general raster / point-cloud writers ------------------------------------
# Not COLMAP-specific: the depth/conf PNG and fused-PLY writers the inference
# entrypoint dumps per sequence. Kept here (rather than a standalone module)
# alongside the other datasets IO, since cv2 + numpy are already imported.
def save_uint16_image(array: np.ndarray, scale: float, path: str) -> None:
    """Scale a float map and write it as a single-channel 16-bit PNG."""
    scaled = np.rint(array.astype(np.float64) * scale)
    scaled = np.clip(scaled, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, scaled)


def write_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a colored point cloud as a binary little-endian PLY."""
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    vertex = np.empty(
        n,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = (
        colors[:, 0],
        colors[:, 1],
        colors[:, 2],
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
