"""Tests for the self-contained COLMAP writers (:mod:`vggt_omega.datasets.utils.colmap_io`).

The writers replicate COLMAP's on-disk formats without depending on colmap/pycolmap, so
the contract under test is *byte compatibility with COLMAP itself*. Two layers:

- Always: pure round-trip against readers written here from the same upstream spec, plus
  the quaternion math and the format's structural invariants.
- When pycolmap is installed: the real oracle -- COLMAP's own reader must load what we
  wrote and return the exact poses/intrinsics/points back. Skipped when absent, so the
  suite still runs in a bare env (mirrors the pycolmap gate in test_sequences.py).
"""

import os
import struct
import tempfile

import numpy as np
import pytest

from vggt_omega.datasets.utils.colmap_io import (
    CAMERA_MODEL_PINHOLE,
    ColmapCamera,
    ColmapImage,
    ColmapPoint3D,
    rotmat_to_qvec,
    write_database,
    write_model_binary,
)


def _pycolmap_available() -> bool:
    try:
        import pycolmap  # noqa: F401

        return True
    except Exception:
        return False


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_xyz(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    return rx @ ry @ _rot_z(az)


def _make_scene(num_frames: int = 3):
    """A tiny synthetic scene: known poses / intrinsics / points (2 points per frame)."""
    cameras, images, points3D, gt = [], [], [], {}
    for i in range(num_frames):
        rot = _rot_xyz(0.1 * i, -0.2 * i, 0.3 * i)
        tvec = np.array([0.1 * i, -0.2 * i, 1.0 + i])
        params = np.array([100.0 + i, 110.0 + i, 32.0, 24.0])
        image_id = i + 1
        cameras.append(
            ColmapCamera(camera_id=image_id, width=64, height=48, params=params)
        )
        images.append(
            ColmapImage(
                image_id=image_id,
                qvec=rotmat_to_qvec(rot),
                tvec=tvec,
                camera_id=image_id,
                name=f"frame_{i:04d}.png",
                xys=np.array([[1.5, 2.5], [3.5, 4.5]]),
                point3D_ids=np.array([2 * i + 1, 2 * i + 2]),
            )
        )
        gt[image_id] = (rot, tvec, params)
        for k in range(2):
            pid = 2 * i + k + 1
            points3D.append(
                ColmapPoint3D(
                    point3D_id=pid,
                    xyz=np.array([0.1 * pid, 0.2 * pid, 0.3 * pid]),
                    rgb=np.array([pid % 256, 7, 9]),
                    error=-1.0,
                    track=[(image_id, k)],
                )
            )
    return cameras, images, points3D, gt


# --------------------------------------------------------------------------- #
# quaternion conversion
# --------------------------------------------------------------------------- #
class TestRotmatToQvec:
    def test_identity(self):
        np.testing.assert_allclose(rotmat_to_qvec(np.eye(3)), [1, 0, 0, 0], atol=1e-12)

    def test_unit_norm_and_positive_w(self):
        for a in (0.0, 0.7, 2.0, 3.0, -1.2):
            q = rotmat_to_qvec(_rot_xyz(a, a / 2, -a))
            assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
            assert q[0] >= 0  # canonical sign

    def test_roundtrips_through_rotation_matrix(self):
        # q -> R must recover the input R for rotations across all Shepperd branches
        # (each branch is selected by which diagonal term dominates).
        def qvec_to_rotmat(q):
            w, x, y, z = q
            return np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                ]
            )

        for rot in (
            np.eye(3),
            _rot_z(np.pi),  # 180 deg -> trace <= 0 branch
            _rot_xyz(np.pi, 0, 0),
            _rot_xyz(0, np.pi, 0),
            _rot_xyz(0.3, -1.1, 2.2),
        ):
            np.testing.assert_allclose(qvec_to_rotmat(rotmat_to_qvec(rot)), rot, atol=1e-9)


# --------------------------------------------------------------------------- #
# binary model: structural checks (no colmap needed)
# --------------------------------------------------------------------------- #
class TestModelBinaryFormat:
    def test_writes_the_three_model_files(self):
        cameras, images, points3D, _ = _make_scene()
        d = tempfile.mkdtemp()
        write_model_binary(d, cameras, images, points3D)
        assert sorted(os.listdir(d)) == ["cameras.bin", "images.bin", "points3D.bin"]

    def test_cameras_bin_header_matches_spec(self):
        # uint64 count, then camera_id:u32, model_id:i32, width:u64, height:u64, params:f64[4]
        cameras, images, points3D, _ = _make_scene(1)
        d = tempfile.mkdtemp()
        write_model_binary(d, cameras, images, points3D)
        with open(os.path.join(d, "cameras.bin"), "rb") as f:
            blob = f.read()
        (count,) = struct.unpack("<Q", blob[:8])
        camera_id, model_id, width, height = struct.unpack("<iiQQ", blob[8:32])
        params = struct.unpack("<4d", blob[32:64])
        assert (count, camera_id, model_id) == (1, 1, CAMERA_MODEL_PINHOLE)
        assert (width, height) == (64, 48)
        np.testing.assert_allclose(params, [100.0, 110.0, 32.0, 24.0])
        assert len(blob) == 64  # exact size => no padding/stray bytes

    def test_points3d_bin_count_and_first_record(self):
        cameras, images, points3D, _ = _make_scene(1)
        d = tempfile.mkdtemp()
        write_model_binary(d, cameras, images, points3D)
        with open(os.path.join(d, "points3D.bin"), "rb") as f:
            blob = f.read()
        (count,) = struct.unpack("<Q", blob[:8])
        assert count == 2
        pid, x, y, z = struct.unpack("<Qddd", blob[8:40])
        r, g, b = struct.unpack("<3B", blob[40:43])
        (error,) = struct.unpack("<d", blob[43:51])
        (track_len,) = struct.unpack("<Q", blob[51:59])
        image_id, point2D_idx = struct.unpack("<ii", blob[59:67])
        assert pid == 1
        np.testing.assert_allclose([x, y, z], [0.1, 0.2, 0.3])
        assert (r, g, b) == (1, 7, 9) and error == -1.0
        assert (track_len, image_id, point2D_idx) == (1, 1, 0)

    def test_image_name_is_null_terminated(self):
        cameras, images, points3D, _ = _make_scene(1)
        d = tempfile.mkdtemp()
        write_model_binary(d, cameras, images, points3D)
        with open(os.path.join(d, "images.bin"), "rb") as f:
            blob = f.read()
        assert b"frame_0000.png\x00" in blob

    def test_negative_point3d_id_maps_to_invalid_sentinel(self):
        # -1 (keypoint with no 3D point) must serialize as COLMAP's uint64 max.
        cameras, _, _, _ = _make_scene(1)
        images = [
            ColmapImage(
                image_id=1,
                qvec=np.array([1.0, 0, 0, 0]),
                tvec=np.zeros(3),
                camera_id=1,
                name="a.png",
                xys=np.array([[1.0, 2.0]]),
                point3D_ids=np.array([-1]),
            )
        ]
        d = tempfile.mkdtemp()
        write_model_binary(d, cameras, images, [])
        with open(os.path.join(d, "images.bin"), "rb") as f:
            blob = f.read()
        assert struct.pack("<Q", 2**64 - 1) in blob


# --------------------------------------------------------------------------- #
# the real oracle: COLMAP's own reader must load what we wrote
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _pycolmap_available(), reason="pycolmap not installed")
class TestAgainstColmapItself:
    def test_model_roundtrips_exactly(self):
        import pycolmap

        cameras, images, points3D, gt = _make_scene(3)
        d = tempfile.mkdtemp()
        model_dir = os.path.join(d, "sparse", "0")
        write_model_binary(model_dir, cameras, images, points3D)

        rec = pycolmap.Reconstruction(model_dir)
        assert rec.num_cameras() == 3
        assert rec.num_images() == 3
        assert rec.num_points3D() == 6

        for image_id, (rot, tvec, params) in gt.items():
            img = rec.image(image_id)
            cam_from_world = img.cam_from_world()
            # poses are world-to-camera in BOTH conventions -> no inversion anywhere
            np.testing.assert_allclose(
                cam_from_world.rotation.matrix(), rot, atol=1e-9
            )
            np.testing.assert_allclose(cam_from_world.translation, tvec, atol=1e-12)
            np.testing.assert_allclose(rec.camera(image_id).params, params, atol=1e-12)
            assert img.name == f"frame_{image_id - 1:04d}.png"
            assert rec.camera(image_id).model.name == "PINHOLE"

        p1 = rec.point3D(1)
        np.testing.assert_allclose(p1.xyz, [0.1, 0.2, 0.3], atol=1e-12)
        assert tuple(p1.color) == (1, 7, 9)

    def test_database_roundtrips_via_native_reader(self):
        import pycolmap

        cameras, images, _, _ = _make_scene(2)
        d = tempfile.mkdtemp()
        db_path = os.path.join(d, "database.db")
        positions = {1: np.array([9.0, 8.0, 7.0]), 2: np.array([-1.0, 0.5, 2.0])}
        write_database(db_path, cameras, images, positions=positions)

        db = pycolmap.Database.open(db_path)
        try:
            assert db.num_cameras() == 2
            assert db.num_images() == 2
            assert db.num_keypoints() == 4  # 2 keypoints x 2 images
            assert db.num_rigs() == 2 and db.num_frames() == 2
            assert db.num_pose_priors() == 2

            cam = db.read_camera(1)
            assert cam.model.name == "PINHOLE" and (cam.width, cam.height) == (64, 48)
            np.testing.assert_allclose(cam.params, [100.0, 110.0, 32.0, 24.0])
            assert db.read_image(1).name == "frame_0000.png"
            np.testing.assert_allclose(
                np.asarray(db.read_keypoints(1))[:, :2], [[1.5, 2.5], [3.5, 4.5]]
            )
            # Current COLMAP keys pose priors by FRAME (is_deprecated_image_prior=False);
            # the default True path deliberately raises on image-keyed lookups.
            for frame_id, position in positions.items():
                assert db.exists_pose_prior(frame_id, False)
                prior = db.read_pose_prior(frame_id, False)
                np.testing.assert_allclose(prior.position, position, atol=1e-12)
                assert prior.coordinate_system.name == "CARTESIAN"
        finally:
            db.close()


# --------------------------------------------------------------------------- #
# dump_as_colmap: the export must be geometrically self-consistent
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _pycolmap_available(), reason="pycolmap not installed")
class TestDumpAsColmapSelfConsistency:
    """Each point3D must reproject onto its OWN recorded 2D observation.

    Regression guard for the half-pixel convention: the world points come from
    ``unproject_depth_map_to_point_map``, which rays through *integer* pixel coords, so
    the stored keypoint must be (x, y) -- storing COLMAP's (x+0.5, y+0.5) pixel centre
    silently puts every observation sqrt(2)/2 = 0.707 px off its own point. A model that
    merely *loads* does not catch this; reprojection does.
    """

    def _tiny_prediction(self, num_frames=3, h=12, w=16):
        rng = np.random.default_rng(7)
        depth = rng.uniform(1.0, 3.0, size=(num_frames, h, w)).astype(np.float32)
        conf = rng.uniform(0.0, 1.0, size=(num_frames, h, w)).astype(np.float32)
        images_hwc = rng.uniform(0, 1, size=(num_frames, h, w, 3)).astype(np.float32)
        extr = np.zeros((num_frames, 3, 4), dtype=np.float64)
        intr = np.zeros((num_frames, 3, 3), dtype=np.float64)
        for i in range(num_frames):
            extr[i, :3, :3] = _rot_xyz(0.05 * i, -0.03 * i, 0.02 * i)
            extr[i, :3, 3] = [0.1 * i, -0.05 * i, 0.02 * i]
            intr[i] = [[20.0 + i, 0, w / 2.0], [0, 21.0 + i, h / 2.0], [0, 0, 1]]
        return images_hwc, depth, conf, extr, intr

    def test_points_reproject_onto_their_observations(self):
        import pycolmap

        from vggt_omega.datasets.utils.colmap_io import dump_as_colmap
        from vggt_omega.utils.geometry import unproject_depth_map_to_point_map

        images_hwc, depth, conf, extr, intr = self._tiny_prediction()
        world = unproject_depth_map_to_point_map(depth[..., None], extr, intr)
        out = tempfile.mkdtemp()
        stats = dump_as_colmap(
            out, images_hwc, depth, conf, extr, intr, world,
            conf_percentile=20.0, max_points=0,  # keep every surviving point
        )
        assert stats["num_points3D"] > 0

        rec = pycolmap.Reconstruction(stats["model_dir"])
        errors = []
        for pid in rec.point3D_ids():
            p = rec.point3D(pid)
            for el in p.track.elements:
                img = rec.image(el.image_id)
                cam = rec.camera(img.camera_id)
                xyz_cam = np.asarray(img.cam_from_world() * p.xyz)
                assert xyz_cam[2] > 0, "point behind its own camera"
                uv = cam.img_from_cam(xyz_cam.reshape(1, 3))[0]
                obs = np.asarray(img.points2D[el.point2D_idx].xy)
                errors.append(np.linalg.norm(uv - obs))
        # Sub-1e-6 px: the observation IS the pixel the point was unprojected from.
        assert np.max(errors) < 1e-6, f"max reprojection error {np.max(errors)} px"

    def test_point_colors_come_from_the_source_pixel(self):
        import pycolmap

        from vggt_omega.datasets.utils.colmap_io import dump_as_colmap
        from vggt_omega.utils.geometry import unproject_depth_map_to_point_map

        images_hwc, depth, conf, extr, intr = self._tiny_prediction()
        world = unproject_depth_map_to_point_map(depth[..., None], extr, intr)
        out = tempfile.mkdtemp()
        stats = dump_as_colmap(
            out, images_hwc, depth, conf, extr, intr, world, max_points=0
        )
        rec = pycolmap.Reconstruction(stats["model_dir"])
        width = images_hwc.shape[2]
        for pid in list(rec.point3D_ids())[:50]:
            p = rec.point3D(pid)
            el = p.track.elements[0]
            obs = np.asarray(rec.image(el.image_id).points2D[el.point2D_idx].xy)
            x, y = int(round(obs[0])), int(round(obs[1]))
            expected = (images_hwc[el.image_id - 1, y, x] * 255).clip(0, 255).astype(np.uint8)
            np.testing.assert_array_equal(np.asarray(p.color), expected)

    def test_max_points_budget_is_respected(self):
        from vggt_omega.datasets.utils.colmap_io import dump_as_colmap
        from vggt_omega.utils.geometry import unproject_depth_map_to_point_map

        images_hwc, depth, conf, extr, intr = self._tiny_prediction()
        world = unproject_depth_map_to_point_map(depth[..., None], extr, intr)
        out = tempfile.mkdtemp()
        stats = dump_as_colmap(
            out, images_hwc, depth, conf, extr, intr, world,
            conf_percentile=0.0, max_points=17,
        )
        assert stats["num_points3D"] == 17


# --------------------------------------------------------------------------- #
# dense workspace: shared synthetic scene with EXACT multi-view consistency
# --------------------------------------------------------------------------- #
def _plane_scene(num_frames=4, h=24, w=32, plane_z=5.0, seed=3):
    """A world plane ``z = plane_z`` seen by slightly rotated/translated cameras.

    Unlike ``_tiny_prediction`` (random depth), the per-frame depth maps here are
    *exactly* multi-view consistent -- unprojecting any frame's pixel and
    reprojecting it into any other frame lands on that frame's own depth. That is
    the property dense track extension validates and ``stereo_fusion`` fuses on,
    so it is what the dense-workspace tests must feed in.
    """
    rng = np.random.default_rng(seed)
    images = rng.uniform(0, 1, size=(num_frames, h, w, 3)).astype(np.float32)
    conf = rng.uniform(0.1, 1.0, size=(num_frames, h, w)).astype(np.float32)
    extr = np.zeros((num_frames, 3, 4), dtype=np.float64)
    intr = np.zeros((num_frames, 3, 3), dtype=np.float64)
    depth = np.zeros((num_frames, h, w), dtype=np.float32)
    fx = fy = 40.0
    cx, cy = w / 2.0, h / 2.0
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    rays = np.stack(
        [(x - cx) / fx, (y - cy) / fy, np.ones((h, w), dtype=np.float64)], axis=-1
    )
    for i in range(num_frames):
        rot = _rot_xyz(0.02 * i, -0.015 * i, 0.01 * i)
        tvec = np.array([0.05 * i, -0.03 * i, 0.02 * i])
        extr[i, :3, :3], extr[i, :3, 3] = rot, tvec
        intr[i] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        cam_center = -rot.T @ tvec
        world_dirs = rays @ rot  # (h, w, 3) = R^T r per pixel
        depth[i] = ((plane_z - cam_center[2]) / world_dirs[..., 2]).astype(np.float32)
    assert (depth > 0).all()
    return images, depth, conf, extr, intr


def _dump_plane_scene(num_frames=4, **dump_kwargs):
    from vggt_omega.datasets.utils.colmap_io import dump_as_colmap
    from vggt_omega.utils.geometry import unproject_depth_map_to_point_map

    images, depth, conf, extr, intr = _plane_scene(num_frames=num_frames)
    world = unproject_depth_map_to_point_map(depth[..., None], extr, intr)
    out = tempfile.mkdtemp()
    dump_kwargs.setdefault("conf_percentile", 20.0)
    dump_kwargs.setdefault("max_points", 0)
    stats = dump_as_colmap(
        out, images, depth, conf, extr, intr, world, **dump_kwargs
    )
    return out, stats, (images, depth, conf, extr, intr, world)


# --------------------------------------------------------------------------- #
# COLMAP mvs Mat<float> maps (depth/normal), format from mvs/mat.cc
# --------------------------------------------------------------------------- #
class TestColmapMatFormat:
    def test_depth_map_bytes_match_colmap_spec(self):
        # mat.cc Write: ASCII "width&height&depth&" then float32 LE payload.
        from vggt_omega.datasets.utils.colmap_io import write_colmap_mat

        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        path = os.path.join(tempfile.mkdtemp(), "d.bin")
        write_colmap_mat(path, arr)
        with open(path, "rb") as f:
            blob = f.read()
        header = b"3&2&1&"
        assert blob[: len(header)] == header
        np.testing.assert_array_equal(
            np.frombuffer(blob[len(header):], dtype="<f4"),
            np.arange(1.0, 7.0, dtype=np.float32),
        )

    def test_multichannel_data_is_slice_major(self):
        # Mat<T>::Get(row, col, slice) = data[slice*W*H + row*W + col]: the file
        # holds channel PLANES (all of c0, then c1, ...), not interleaved pixels.
        from vggt_omega.datasets.utils.colmap_io import write_colmap_mat

        h, w = 2, 3
        arr = np.zeros((h, w, 3), dtype=np.float32)
        for c in range(3):
            arr[..., c] = np.arange(h * w, dtype=np.float32).reshape(h, w) + 100 * c
        path = os.path.join(tempfile.mkdtemp(), "n.bin")
        write_colmap_mat(path, arr)
        with open(path, "rb") as f:
            blob = f.read()
        assert blob[:6] == b"3&2&3&"
        planes = np.frombuffer(blob[6:], dtype="<f4").reshape(3, h, w)
        for c in range(3):
            np.testing.assert_array_equal(planes[c], arr[..., c])

    def test_roundtrip_single_and_three_channel(self):
        from vggt_omega.datasets.utils.colmap_io import (
            read_colmap_mat,
            write_colmap_mat,
        )

        rng = np.random.default_rng(0)
        d = tempfile.mkdtemp()
        for shape in ((5, 7), (4, 6, 3)):
            arr = rng.uniform(-2, 9, size=shape).astype(np.float32)
            path = os.path.join(d, f"m{len(shape)}.bin")
            write_colmap_mat(path, arr)
            back = read_colmap_mat(path)
            expected = arr if arr.ndim == 3 else arr[..., None]
            assert back.shape == expected.shape and back.dtype == np.float32
            np.testing.assert_array_equal(back, expected)


# --------------------------------------------------------------------------- #
# normals from depth (camera frame; what the dense normal maps carry)
# --------------------------------------------------------------------------- #
class TestDepthMapToCameraNormals:
    def _intr(self, s, fx, fy, cx, cy):
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        return np.tile(K, (s, 1, 1))

    def test_frontoparallel_plane_gives_minus_z(self):
        # Constant depth = plane orthogonal to the optical axis; COLMAP's convention
        # is normals in the camera frame pointing AT the camera -> (0, 0, -1).
        from vggt_omega.datasets.utils.colmap_io import fake_depth_map_to_camera_normals

        depth = np.full((1, 10, 14), 2.5, dtype=np.float32)
        normals = fake_depth_map_to_camera_normals(
            depth, self._intr(1, 30.0, 31.0, 7.0, 5.0)
        )
        assert normals.shape == (1, 10, 14, 3)
        np.testing.assert_allclose(
            normals, np.broadcast_to([0.0, 0.0, -1.0], normals.shape), atol=1e-6
        )

    def test_slanted_plane_recovers_its_normal(self):
        from vggt_omega.datasets.utils.colmap_io import fake_depth_map_to_camera_normals

        h, w = 24, 32
        fx = fy = 40.0
        cx, cy = w / 2.0, h / 2.0
        n_true = np.array([0.3, -0.2, -0.9])
        n_true /= np.linalg.norm(n_true)
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        rays = np.stack(
            [(x - cx) / fx, (y - cy) / fy, np.ones((h, w))], axis=-1
        )
        # Plane n . P = -2 in camera coords, P = depth * ray.
        depth = (-2.0 / (rays @ n_true)).astype(np.float32)[None]
        assert (depth > 0).all()
        normals = fake_depth_map_to_camera_normals(depth, self._intr(1, fx, fy, cx, cy))
        interior = normals[0, 2:-2, 2:-2]
        assert (interior @ n_true).min() > 0.999
        # Camera-facing everywhere: n . P < 0.
        P = rays * depth[0][..., None]
        assert (np.einsum("hwc,hwc->hw", normals[0], P) < 0).all()

    def test_invalid_depth_gives_zero_normals_valid_stays_unit(self):
        from vggt_omega.datasets.utils.colmap_io import fake_depth_map_to_camera_normals

        depth = np.full((1, 12, 12), 3.0, dtype=np.float32)
        depth[0, 4:7, 5:8] = 0.0
        depth[0, 1, 1] = np.nan
        normals = fake_depth_map_to_camera_normals(
            depth, self._intr(1, 25.0, 25.0, 6.0, 6.0)
        )
        invalid = ~(np.isfinite(depth[0]) & (depth[0] > 0))
        assert (normals[0][invalid] == 0).all()
        np.testing.assert_allclose(
            np.linalg.norm(normals[0][~invalid], axis=-1), 1.0, atol=1e-5
        )


# --------------------------------------------------------------------------- #
# dump_as_colmap: dense workspace layout + contents (no pycolmap needed)
# --------------------------------------------------------------------------- #
class TestDumpAsColmapDenseWorkspace:
    def test_dense_layout_and_cfgs(self):
        out, stats, (images, depth, conf, extr, intr, world) = _dump_plane_scene()
        dense = os.path.join(out, "colmap", "dense")
        assert stats["dense_dir"] == dense
        assert stats["num_dense_points3D"] > 0
        for sub in (
            "images",
            "sparse",
            "stereo/depth_maps",
            "stereo/normal_maps",
            "stereo/consistency_graphs",
        ):
            assert os.path.isdir(os.path.join(dense, sub)), sub
        for fn in ("cameras.bin", "images.bin", "points3D.bin"):
            assert os.path.isfile(os.path.join(dense, "sparse", fn)), fn

        names = [f"frame_{i:04d}.png" for i in range(depth.shape[0])]
        with open(os.path.join(dense, "stereo", "fusion.cfg")) as f:
            assert f.read().split() == names
        with open(os.path.join(dense, "stereo", "patch-match.cfg")) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert lines[0::2] == names
        assert all(line == "__auto__, 20" for line in lines[1::2])
        for name in names:
            assert os.path.isfile(os.path.join(dense, "images", name))
            for kind in ("depth_maps", "normal_maps"):
                assert os.path.isfile(
                    os.path.join(dense, "stereo", kind, name + ".geometric.bin")
                )

    def test_depth_maps_are_conf_masked_predictions(self):
        from vggt_omega.datasets.utils.colmap_io import read_colmap_mat

        out, stats, (images, depth, conf, extr, intr, world) = _dump_plane_scene()
        dense = stats["dense_dir"]
        for i in range(depth.shape[0]):
            got = read_colmap_mat(
                os.path.join(
                    dense, "stereo", "depth_maps", f"frame_{i:04d}.png.geometric.bin"
                )
            )[..., 0]
            keep = (
                np.isfinite(world[i]).all(axis=-1)
                & np.isfinite(conf[i])
                & (depth[i] > 0)
            )
            keep &= conf[i] >= np.percentile(conf[i][keep], 20.0)
            expected = np.where(keep, depth[i], 0.0).astype(np.float32)
            np.testing.assert_array_equal(got, expected)

    def test_images_are_the_predicted_rgb(self):
        import cv2

        out, stats, (images, depth, conf, extr, intr, world) = _dump_plane_scene()
        bgr = cv2.imread(os.path.join(stats["dense_dir"], "images", "frame_0000.png"))
        expected = (images[0] * 255.0).clip(0, 255).astype(np.uint8)[..., ::-1]
        np.testing.assert_array_equal(bgr, expected)

    def test_normal_maps_match_plane_normal_and_depth_mask(self):
        from vggt_omega.datasets.utils.colmap_io import read_colmap_mat

        out, stats, (images, depth, conf, extr, intr, world) = _dump_plane_scene()
        dense = stats["dense_dir"]
        i = 1
        normals = read_colmap_mat(
            os.path.join(
                dense, "stereo", "normal_maps", f"frame_{i:04d}.png.geometric.bin"
            )
        )
        written_depth = read_colmap_mat(
            os.path.join(
                dense, "stereo", "depth_maps", f"frame_{i:04d}.png.geometric.bin"
            )
        )[..., 0]
        valid = written_depth > 0
        # World plane z=Z0 faces the cameras with world normal (0,0,-1); in the
        # camera frame that is R @ (0,0,-1).
        expected = extr[i][:3, :3] @ np.array([0.0, 0.0, -1.0])
        interior = valid.copy()
        interior[:2, :] = interior[-2:, :] = False
        interior[:, :2] = interior[:, -2:] = False
        assert (normals[interior] @ expected).min() > 0.999
        assert (normals[~valid] == 0).all()
        np.testing.assert_allclose(
            np.linalg.norm(normals[valid], axis=-1), 1.0, atol=1e-5
        )


# --------------------------------------------------------------------------- #
# dense sparse model: cross-view tracks (what gives stereo_fusion its overlap
# graph -- single-image tracks make it fuse nothing)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _pycolmap_available(), reason="pycolmap not installed")
class TestDenseSparseModelWithPycolmap:
    def test_cross_view_tracks_exist_and_reproject_exactly(self):
        import pycolmap

        out, stats, (images, depth, conf, extr, intr, world) = _dump_plane_scene(
            dense_tracks_per_frame=64
        )
        rec = pycolmap.Reconstruction(os.path.join(stats["dense_dir"], "sparse"))
        assert rec.num_images() == depth.shape[0]

        track_lens = [
            len(rec.point3D(pid).track.elements) for pid in rec.point3D_ids()
        ]
        # The scene is exactly consistent, so most tracks must span >= 2 views and
        # some point must be validated in EVERY view.
        assert max(track_lens) == depth.shape[0]
        assert np.mean(np.asarray(track_lens) >= 2) > 0.5

        errors = []
        for pid in rec.point3D_ids():
            p = rec.point3D(pid)
            for el in p.track.elements:
                img = rec.image(el.image_id)
                cam = rec.camera(img.camera_id)
                xyz_cam = np.asarray(img.cam_from_world() * p.xyz)
                assert xyz_cam[2] > 0
                uv = cam.img_from_cam(xyz_cam.reshape(1, 3))[0]
                obs = np.asarray(img.points2D[el.point2D_idx].xy)
                errors.append(np.linalg.norm(uv - obs))
        assert np.max(errors) < 1e-4, f"max reprojection error {np.max(errors)} px"

    def test_dense_model_poses_and_cameras_match_predictions(self):
        import pycolmap

        out, stats, (images, depth, conf, extr, intr, world) = _dump_plane_scene()
        rec = pycolmap.Reconstruction(os.path.join(stats["dense_dir"], "sparse"))
        for i in range(depth.shape[0]):
            img = rec.image(i + 1)
            cam_from_world = img.cam_from_world()
            np.testing.assert_allclose(
                cam_from_world.rotation.matrix(), extr[i][:3, :3], atol=1e-9
            )
            np.testing.assert_allclose(
                cam_from_world.translation, extr[i][:3, 3], atol=1e-12
            )
            assert rec.camera(img.camera_id).model.name == "PINHOLE"

    def test_tracks_link_each_frame_to_its_nearest_views(self):
        """The dense tracks must make NEARBY views the strongest overlaps.

        stereo_fusion traverses only ``model.GetMaxOverlappingImages(...)``, which
        ranks images purely by shared-track count (``mvs/model.cc
        ComputeSharedPoints``). If a track spreads its views evenly across the whole
        sequence, an image's strongest "overlaps" become far-away viewpoints and its
        immediate neighbours drop out of the list -- fusion then gathers support only
        where the predicted depth agrees worst, every point falls under
        ``min_num_pixels``, and the cloud collapses. Measured on real garden output
        before this was fixed: neighbours present in the top-50 just 0.5% of the time,
        80% of fused points left with support 1.

        A model that merely *loads* does not catch this; only the overlap ranking does.
        """
        import pycolmap

        out, stats, (images, depth, conf, extr, intr, world) = _dump_plane_scene(
            num_frames=8, dense_tracks_per_frame=48, dense_track_max_length=4
        )
        rec = pycolmap.Reconstruction(os.path.join(stats["dense_dir"], "sparse"))
        ids = sorted(rec.reg_image_ids())
        index = {im: k for k, im in enumerate(ids)}

        shared = {}
        for pid in rec.point3D_ids():
            els = [index[e.image_id] for e in rec.point3D(pid).track.elements]
            for a in range(len(els)):
                for b in range(a):
                    for key in ((els[a], els[b]), (els[b], els[a])):
                        shared[key] = shared.get(key, 0) + 1

        # Every point here is visible in all 8 frames, so the 4-view cap must choose.
        # It has to choose the NEAREST views, or fusion's traversal targets are wrong.
        for i in range(len(ids)):
            ranked = sorted(
                ((j, c) for (a, j), c in shared.items() if a == i),
                key=lambda kv: -kv[1],
            )
            top = [j for j, _ in ranked[:3]]
            neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(ids)]
            assert all(nb in top for nb in neighbours), (
                f"image {i}: immediate neighbours {neighbours} missing from strongest "
                f"overlaps {top} -- fusion cannot reach them"
            )

    def test_workspace_is_complete_for_every_model_image(self):
        # The contract fusion.cc checks per image before using it: the name is in
        # the model, and bitmap / depth map / normal map files all exist. A name
        # mismatch (or missing map) makes COLMAP silently ignore the image, so the
        # dump must guarantee cfg names == model names == files on disk.
        import pycolmap

        out, stats, (images, depth, conf, extr, intr, world) = _dump_plane_scene()
        dense = stats["dense_dir"]
        rec = pycolmap.Reconstruction(os.path.join(dense, "sparse"))
        model_names = sorted(img.name for img in rec.images.values())
        with open(os.path.join(dense, "stereo", "fusion.cfg")) as f:
            cfg_names = sorted(f.read().split())
        assert cfg_names == model_names
        for name in model_names:
            assert os.path.isfile(os.path.join(dense, "images", name))
            for kind in ("depth_maps", "normal_maps"):
                assert os.path.isfile(
                    os.path.join(dense, "stereo", kind, name + ".geometric.bin")
                )
