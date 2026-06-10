import numpy as np
import torch

from vggt_omega.utils.geometry import project_world_points_to_cam, cam_from_img, sampson_epipolar_distance
from vggt_omega.datasets.composed_dataset import ComposedDataset
from vggt_omega.datasets.track_util import build_tracks_by_depth, track_epipolar_check


def _two_cam_scene():
    # cam0 at origin looking +z; cam1 translated +x by 0.5, same orientation; K fx=fy=100, pp=(32,32), 64x64
    K = torch.tensor([[100., 0, 32], [0, 100., 32], [0, 0, 1]]).expand(2, 3, 3).clone()
    E = torch.zeros(2, 3, 4); E[:, :3, :3] = torch.eye(3)
    E[1, 0, 3] = -0.5  # w2c: x_cam = R x_world + t, so t = -0.5 shifts world +x into view
    return E, K


def test_project_world_points_roundtrip():
    E, K = _two_cam_scene()
    pts = torch.tensor([[0.0, 0.0, 2.0], [0.3, -0.2, 4.0]])  # world points in front of both cams
    img_pts, cam_pts = project_world_points_to_cam(pts, E, K)
    assert img_pts.shape == (2, 2, 2) and cam_pts.shape == (2, 3, 2)
    # cam0 == world: pixel = K @ (x/z, y/z, 1)
    assert torch.allclose(img_pts[0, 0], torch.tensor([32.0, 32.0]), atol=1e-5)
    assert torch.allclose(cam_pts[0, :, 1], pts[1], atol=1e-6)
    # depth row is z
    assert torch.allclose(cam_pts[1, 2], torch.tensor([2.0, 4.0]), atol=1e-6)


def test_cam_from_img_normalizes():
    E, K = _two_cam_scene()
    tracks = torch.tensor([[[32.0, 32.0], [132.0, 32.0]]]).expand(2, 2, 2).clone()
    norm = cam_from_img(tracks, K)
    assert torch.allclose(norm[0, 0], torch.tensor([0.0, 0.0]), atol=1e-6)
    assert torch.allclose(norm[0, 1], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_sampson_distance_epipolar_points_are_zero():
    E, K = _two_cam_scene()
    pts = torch.tensor([[0.0, 0.0, 2.0], [0.3, -0.2, 4.0], [-0.1, 0.25, 3.0]])
    img_pts, _ = project_world_points_to_cam(pts, E, K)
    d = track_epipolar_check(img_pts, E, K)   # (1, P) sampson distances frame0 -> frame1
    assert d.shape == (1, 3)
    assert torch.all(d.abs() < 1e-3)          # true correspondences lie on epipolar lines


def test_sampson_distance_off_epipolar_is_large():
    E, K = _two_cam_scene()
    pts = torch.tensor([[0.0, 0.0, 2.0]])
    img_pts, _ = project_world_points_to_cam(pts, E, K)
    img_pts[1, :, 1] += 20.0  # cameras differ in x only -> epipolar lines horizontal; +20px in y is off-line
    d = track_epipolar_check(img_pts, E, K)
    assert torch.all(d > 16.0)


def test_build_tracks_by_depth_end_to_end():
    # 2 frames, 64x64, flat plane z=2 visible in both
    E, K = _two_cam_scene()
    # smaller baseline than the geometry tests: sample_positive_tracks keeps the
    # top HALF of all query pixels by valid-frame count, so co-visible tracks must
    # outnumber half the pool for every sampled positive to be query-visible
    E[1, 0, 3] = -0.1
    H = W = 64
    vs, us = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    z = torch.full((H, W), 2.0)
    x = (us - 32) / 100 * z; y = (vs - 32) / 100 * z
    wp0 = torch.stack([x, y, z], -1)                       # cam0 == world
    wp1 = wp0.clone(); wp1[..., 0] += 0.0                  # static scene, same world points
    world_points = torch.stack([wp0, wp1])                 # (2,H,W,3) — frame1's stored points are its own unprojection;
    depths = torch.stack([z, z + 0.0])                     # for a flat fronto-parallel plane depth is z in both cams
    masks = torch.ones(2, H, W, dtype=torch.bool)
    images = torch.zeros(2, 3, H, W)
    tracks, vis, pos = build_tracks_by_depth(E, K, world_points, depths, masks, images,
                                             target_track_num=128, neg_ratio=0.25)
    assert tracks.shape == (2, 128, 2) and vis.shape == (2, 128) and pos.shape == (128,)
    assert pos.sum() >= 64                                  # plenty of positives on a fully-valid plane
    assert vis[0, pos].all()                                # positives visible in query frame
    assert not vis[:, ~pos].any()                           # negatives carry vis=False by contract


def _stub_vendor_batch():
    # raw numpy batch as a vendor's get_data would return it, WITHOUT a 'tracks' key
    H = W = 64
    K = np.tile(np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1]], dtype=np.float32), (2, 1, 1))
    E = np.zeros((2, 3, 4), dtype=np.float32); E[:, :3, :3] = np.eye(3)
    E[1, 0, 3] = -0.1
    vs, us = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    z = np.full((H, W), 2.0, dtype=np.float32)
    x = (us - 32) / 100 * z; y = (vs - 32) / 100 * z
    wp = np.stack([x, y, z], -1)
    cam_points = np.stack([wp, wp - np.array([0.1, 0, 0], dtype=np.float32)])
    return {
        "seq_name": "stub_plane",
        "ids": np.array([0, 1]),
        "images": np.zeros((2, H, W, 3), dtype=np.uint8),
        "depths": np.stack([z, z]),
        "extrinsics": E,
        "intrinsics": K,
        "cam_points": cam_points,
        "world_points": np.stack([wp, wp]),
        "point_masks": np.ones((2, H, W), dtype=bool),
    }


def test_tensorize_builds_tracks_when_vendor_has_none():
    ds = object.__new__(ComposedDataset)
    ds.load_track = True
    ds.track_num = 64
    ds.track_neg_ratio = 0.25
    ds.training = False
    sample = ds._tensorize(_stub_vendor_batch())
    assert sample["tracks"].shape == (2, 64, 2)
    assert sample["track_vis_mask"].dtype == torch.bool
    assert sample["track_positive_mask"].shape == (64,)
    # neg_ratio plumbed through: 64 - int(64 * 0.25) positive slots
    assert sample["track_positive_mask"].sum() == 48


def test_build_tracks_unfilled_slots_are_out_of_bounds():
    # When no negatives survive the epipolar check, leftover slots must NOT
    # alias pixel (0, 0) — zeros would create false negative pairs at the
    # top-left patch in a matching loss that selects negatives by in-bounds coords.
    E, K = _two_cam_scene()
    E[1, 0, 3] = -0.1
    H = W = 64
    vs, us = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    z = torch.full((H, W), 2.0)
    x = (us - 32) / 100 * z; y = (vs - 32) / 100 * z
    wp0 = torch.stack([x, y, z], -1)
    world_points = torch.stack([wp0, wp0.clone()])
    depths = torch.stack([z, z + 0.0])
    masks = torch.ones(2, H, W, dtype=torch.bool)
    images = torch.zeros(2, 3, H, W)
    tracks, vis, pos = build_tracks_by_depth(E, K, world_points, depths, masks, images,
                                             target_track_num=128, neg_ratio=0.5,
                                             neg_epipolar_thres=1e12)
    assert (~pos).any()
    assert (tracks[:, ~pos] < 0).all()
