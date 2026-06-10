import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.kitti import KittiDataset

KITTI_DIR = "/jfs/guibiao/streamVGGT/data/eval/kitti"
HAVE_KITTI = os.path.isdir(KITTI_DIR)
# Smallest val drive (67 GT-depth frames per camera); used for fast integration tests.
DRIVE = "2011_09_26_drive_0002_sync"


def _common_conf():
    return OmegaConf.create(
        {
            "img_size": 512,
            "patch_size": 16,
            "training": True,
            "inside_random": False,
            "allow_duplicate_img": True,
            "get_nearby": True,
            "rescale": True,
            "rescale_aug": True,
            "landscape_check": False,
            "augs": {"scales": [0.8, 1.2]},
        }
    )


def _integration_common():
    return OmegaConf.merge(
        _common_conf(),
        OmegaConf.create(
            {
                "fix_img_num": -1,
                "fix_aspect_ratio": 1.0,
                "load_track": False,
                "track_num": 1024,
                "load_depth": True,
                "debug": False,
                "repeat_batch": False,
                "img_nums": [2, 6],
                "max_img_per_gpu": 12,
                "augs": {
                    "scales": [0.8, 1.2],
                    "aspects": [1.0, 1.0],
                    "cojitter": False,
                    "cojitter_ratio": 0.3,
                    "color_jitter": None,
                    "gray_scale": False,
                    "gau_blur": False,
                },
            }
        ),
    )


def _eval_common():
    """Deterministic eval-mode common_config: no aug, no random remap, explicit ids
    honored verbatim (matches how inference.py drives the loader)."""
    return OmegaConf.merge(
        _integration_common(),
        OmegaConf.create(
            {
                "training": False,
                "inside_random": False,
                "rescale_aug": False,
                "get_nearby": False,
                "allow_duplicate_img": False,
                "augs": {"scales": None},
            }
        ),
    )


def _kitti_dataset_cfg(seqs=(DRIVE,), n=20, cameras=(2, 3)):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.kitti.KittiDataset",
                "split": "val",
                "KITTI_DIR": KITTI_DIR,
                "sequences": list(seqs),
                "cameras": list(cameras),
                "len_test": n,
            }
        ],
    }


# --- KITTI-specific helper unit tests (no data required) ---


def test_rotation_from_rpy_identity_and_yaw():
    np.testing.assert_allclose(KittiDataset.rotation_from_rpy(0.0, 0.0, 0.0), np.eye(3), atol=1e-12)
    # yaw = +90 deg (CCW about z): x-axis -> y-axis
    R = KittiDataset.rotation_from_rpy(0.0, 0.0, np.pi / 2)
    np.testing.assert_allclose(R @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)  # orthonormal


def test_mercator_scale():
    assert KittiDataset.mercator_scale(0.0) == pytest.approx(1.0)
    assert KittiDataset.mercator_scale(60.0) == pytest.approx(0.5)


def test_oxts_to_imu_pose_origin_and_translation():
    # lat=lon=alt=0, rpy=0 -> identity pose (mercator origin)
    pose = KittiDataset.oxts_to_imu_pose([0.0] * 6, scale=1.0)
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-9)
    # longitude moves +x by scale * lon_rad * earth_radius
    er = KittiDataset._EARTH_RADIUS
    lon = 100.0 / er * 180.0 / np.pi  # -> tx = 100 m at scale 1
    pose = KittiDataset.oxts_to_imu_pose([0.0, lon, 7.0, 0.0, 0.0, 0.0], scale=1.0)
    np.testing.assert_allclose(pose[:3, 3], [100.0, 0.0, 7.0], atol=1e-6)


def test_oxts_to_imu_pose_rejects_bad_records():
    with pytest.raises(ValueError, match=">= 6"):
        KittiDataset.oxts_to_imu_pose([1.0, 2.0, 3.0], scale=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        KittiDataset.oxts_to_imu_pose([np.nan] * 6, scale=1.0)


def test_cam_from_imu_rigid_and_stereo_baseline():
    t2 = KittiDataset.cam_from_imu("2011_09_26", 2)
    t3 = KittiDataset.cam_from_imu("2011_09_26", 3)
    assert t2.shape == (4, 4)
    np.testing.assert_allclose(t2[:3, :3] @ t2[:3, :3].T, np.eye(3), atol=1e-5)  # rigid
    np.testing.assert_allclose(t2[3], [0, 0, 0, 1], atol=1e-12)
    # cam3 differs from cam2 only by the rectified stereo baseline along cam-x:
    # T3 @ inv(T2) translation = (P_rect_03[0,3] - P_rect_02[0,3]) / fx
    rel = t3 @ np.linalg.inv(t2)
    expected_bx = (-3.395242e02 - 4.485728e01) / 7.215377e02
    np.testing.assert_allclose(rel[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(rel[:3, 3], [expected_bx, 0.0, 0.0], atol=1e-9)


def test_cam_from_imu_rejects_unknown():
    with pytest.raises(ValueError, match="calib"):
        KittiDataset.cam_from_imu("2099_01_01", 2)
    with pytest.raises(ValueError, match="cam"):
        KittiDataset.cam_from_imu("2011_09_26", 1)


def test_kitti_intrinsics_per_date_override_and_unknown():
    K = KittiDataset.kitti_intrinsics("2011_09_26", 2)
    assert K.shape == (3, 3) and K.dtype == np.float32
    assert K[0, 0] == pytest.approx(721.5377) and K[1, 1] == pytest.approx(721.5377)
    assert K[0, 2] == pytest.approx(609.5593) and K[1, 2] == pytest.approx(172.854)
    # cams 02/03 share K by rectification design
    np.testing.assert_allclose(K, KittiDataset.kitti_intrinsics("2011_09_26", 3))
    K2 = KittiDataset.kitti_intrinsics("anything", override=[100.0, 100.0, 50.0, 40.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0 and K2[1, 2] == 40.0
    with pytest.raises(ValueError, match="calib"):
        KittiDataset.kitti_intrinsics("2099_01_01", 2)


def test_imu_pose_to_w2c_identity_anchor_and_nonfinite():
    w2c = KittiDataset.imu_pose_to_w2c(np.eye(4), np.eye(4))
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-7)
    # recentering: anchor subtracts the IMU position before inversion
    t_w_imu = np.eye(4)
    t_w_imu[:3, 3] = [1e6, 2e6, 3.0]
    w2c = KittiDataset.imu_pose_to_w2c(t_w_imu, np.eye(4), anchor=[1e6, 2e6, 0.0])
    np.testing.assert_allclose(w2c[:, 3], [0.0, 0.0, -3.0], atol=1e-7)
    bad = np.eye(4)
    bad[0, 3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        KittiDataset.imu_pose_to_w2c(bad, np.eye(4))


def test_read_kitti_depth_units_and_invalid(tmp_path):
    import cv2

    arr = np.array([[0, 256], [512, 25600]], dtype=np.uint16)
    p = str(tmp_path / "0000000005.png")
    assert cv2.imwrite(p, arr)
    depth = KittiDataset.read_kitti_depth(p)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[0.0, 1.0], [2.0, 100.0]])  # /256 -> m, 0 invalid
    with pytest.raises(FileNotFoundError):
        KittiDataset.read_kitti_depth(str(tmp_path / "missing.png"))


def test_parse_kitti_timestamp():
    assert KittiDataset.parse_kitti_timestamp("1970-01-01 00:00:01.500000000") == pytest.approx(1.5)
    t1 = KittiDataset.parse_kitti_timestamp("2011-09-26 13:02:44.335092332")
    t2 = KittiDataset.parse_kitti_timestamp("2011-09-26 13:02:44.435100758")
    assert t2 - t1 == pytest.approx(0.100008426, abs=1e-6)
    with pytest.raises(ValueError):
        KittiDataset.parse_kitti_timestamp("   ")


def test_parse_gt_depth_listing_and_drive_date():
    files = ["0000000010.png", "0000000005.png", "junk.txt", "5.png", "0000000007.png"]
    assert KittiDataset.parse_gt_depth_listing(files) == [5, 7, 10]
    assert KittiDataset.drive_date("2011_09_26_drive_0002_sync") == "2011_09_26"


def test_available_modalities_follow_sparse_lidar_rules():
    # Sparse LiDAR: sky is indistinguishable from missing returns -> no SKY_MASK;
    # reprojected points are not point-cloud GT -> no WORLD_POINTS / CAM_POINTS.
    assert Modality.SKY_MASK not in KittiDataset.AVAILABLE
    assert Modality.WORLD_POINTS not in KittiDataset.AVAILABLE
    assert Modality.CAM_POINTS not in KittiDataset.AVAILABLE
    assert Modality.TIMESTAMP in KittiDataset.AVAILABLE
    assert Modality.CAMERA_ID in KittiDataset.AVAILABLE


# --- KITTI integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_kitti_sample_schema_and_conventions():
    ds = KittiDataset(
        common_conf=_common_conf(),
        split="val",
        KITTI_DIR=KITTI_DIR,
        sequences=[DRIVE],
        len_test=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert ds.sequence_list == [f"{DRIVE}/cam02", f"{DRIVE}/cam03"]

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    img = np.stack(batch["images"])
    depth = np.stack(batch["depths"])
    extr = np.stack(batch["extrinsics"])
    intr = np.stack(batch["intrinsics"])
    world = np.stack(batch["world_points"])
    pmask = np.stack(batch["point_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    # recentered per drive: translations stay small enough for float32
    assert np.abs(extr[:, :, 3]).max() < 1e4
    assert (depth[depth > 0]).size > 0                    # some valid (sparse) depth
    assert (depth >= 0).all()                             # no sky encoding in LiDAR GT
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is True
    assert "sky_masks" not in batch                       # sparse LiDAR: no sky GT
    assert batch["camera_ids"].dtype == np.int32
    assert set(batch["camera_ids"].tolist()) == {2}       # seq 0 is the cam02 stream
    assert batch["timestamps"].dtype == np.float64

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_kitti_cameras_knob_and_camera_ids():
    ds = KittiDataset(
        common_conf=_eval_common(),
        split="val",
        KITTI_DIR=KITTI_DIR,
        sequences=[DRIVE],
        cameras=(3,),
        len_test=10,
    )
    assert ds.sequence_list == [f"{DRIVE}/cam03"]
    batch = ds.get_data(seq_name=ds.sequence_list[0], ids=np.array([0, 1]), aspect_ratio=1.0)
    assert set(batch["camera_ids"].tolist()) == {3}
    with pytest.raises(ValueError, match="cameras"):
        KittiDataset(
            common_conf=_eval_common(), split="val", KITTI_DIR=KITTI_DIR,
            sequences=[DRIVE], cameras=(0,),
        )


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_kitti_timestamps_are_real_and_ordered():
    """Timestamps come from the on-disk per-camera capture clock (~10 Hz);
    sorted ids must yield strictly increasing times with ~0.1 s spacing."""
    ds = KittiDataset(
        common_conf=_eval_common(),
        split="val",
        KITTI_DIR=KITTI_DIR,
        sequences=[DRIVE],
        len_test=10,
    )
    batch = ds.get_data(seq_name=f"{DRIVE}/cam02", ids=np.array([0, 1, 2, 10]), aspect_ratio=1.0)
    ts = batch["timestamps"]
    assert (np.diff(ts) > 0).all()
    assert np.diff(ts)[:2] == pytest.approx([0.1, 0.1], abs=0.05)   # consecutive frames
    assert ts[3] - ts[0] == pytest.approx(1.0, abs=0.3)             # 10 frames at ~10 Hz


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_kitti_reprojection_closure():
    """World points from two frames of the same sequence must be mutually
    consistent: frame A's valid world points reprojected into frame B land at
    depths matching B's sparse LiDAR map (validates the oxts->cam pose chain +
    hardcoded devkit calib end-to-end through process_one_image). The flipped
    (c2w-as-w2c) convention measures ~0.45 median relative error on this drive
    pair (4.5 m baseline), so the 0.03 threshold fails hard under a flip."""
    ds = KittiDataset(
        common_conf=_eval_common(),
        split="val",
        KITTI_DIR=KITTI_DIR,
        sequences=[DRIVE],
        len_test=10,
    )
    # ids 0 and 5 -> drive frames 5 and 10 (GT depth starts at frame 5): 4.5 m apart
    b = ds.get_data(seq_name=f"{DRIVE}/cam02", ids=np.array([0, 5]), aspect_ratio=0.3)
    world = np.stack(b["world_points"])
    pmask = np.stack(b["point_masks"])
    extr = np.stack(b["extrinsics"])
    intr = np.stack(b["intrinsics"])
    depth = np.stack(b["depths"])

    wA = world[0][pmask[0]]
    E, K = extr[1], intr[1]
    camB = wA @ E[:3, :3].T + E[:3, 3]
    z = camB[:, 2]
    u = camB[:, 0] / z * K[0, 0] + K[0, 2]
    v = camB[:, 1] / z * K[1, 1] + K[1, 2]
    H, W = depth[1].shape
    ok = (z > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.round(u[ok]).astype(int)
    vi = np.round(v[ok]).astype(int)
    measured = depth[1][vi, ui]
    valid = measured > 0
    rel_err = np.abs(z[ok][valid] - measured[valid]) / measured[valid]
    assert valid.sum() >= 500
    assert np.median(rel_err) < 0.03


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_kitti_getitem_tuple_index():
    ds = KittiDataset(
        common_conf=_common_conf(),
        split="val",
        KITTI_DIR=KITTI_DIR,
        sequences=[DRIVE],
        len_test=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_kitti_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _kitti_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "timestamps" in sample                  # extended modality carried through
    assert "camera_ids" in sample
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _kitti_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]
    sample = composed.get_sample(0, ids=ids, aspect_ratio=0.75)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample

    # Drift guard: the same vendor.get_data + manual tensorize must match byte-for-byte.
    vendor = composed.base_dataset.datasets[0]
    batch = vendor.get_data(
        seq_name=composed.sequence_name(0), ids=np.array(ids), aspect_ratio=0.75
    )
    manual = (
        torch.from_numpy(np.stack(batch["images"]).astype(np.float32))
        .permute(0, 3, 1, 2)
        .to(torch.get_default_dtype())
        .div(255)
    )
    torch.testing.assert_close(sample["images"], manual)
    # Order honored: per-frame timestamps follow the requested id order.
    np.testing.assert_allclose(sample["timestamps"].numpy(), batch["timestamps"])


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_test), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    composed = instantiate(
        _kitti_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    vendor = composed.base_dataset.datasets[0]
    assert composed.num_sequences() == 2               # one drive x cams {02, 03}
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in vendor.sequence_list
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name])
        assert composed.sequence_num_frames(gi) == 67  # GT depth frames 5..71


@pytest.mark.skipif(not HAVE_KITTI, reason=f"KITTI data not found at {KITTI_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _kitti_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (375, 1242)                      # 2011_09_26 native (H, W)

    composed.set_img_size(1232)                       # ~native long side, /16-friendly
    assert composed.img_size == 1232
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (368, 1232)

    composed.set_img_size(624)                        # ~half-res long side
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (176, 624)
