import logging
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.waymo import WaymoDataset

WAYMO_DIR = "/jfs/Data_4DFF/train_data/waymo"
HAVE_WAYMO = os.path.isdir(WAYMO_DIR)
# One verified segment (the survey's gold-check segment); restricting `sequences`
# to it keeps integration tests fast. cam1 = FRONT (341x512), cam4 = SIDE_LEFT (236x512).
SEGMENT = "segment-10017090168044687777_6380_000_6400_000"
SEQ_CAM1 = f"{SEGMENT}/cam1"
SEQ_CAM4 = f"{SEGMENT}/cam4"


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


def _waymo_dataset_cfg(seqs=(SEQ_CAM1,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.waymo.WaymoDataset",
                "split": "train",
                "WAYMO_DIR": WAYMO_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- Waymo-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = WaymoDataset.waymo_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_rotation_is_rigid_inverse():
    # 90 deg about z (OpenCV axes, no axis remap): w2c composed with c2w = identity
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [4.0, -5.0, 6.0]
    w2c = WaymoDataset.waymo_pose_to_w2c(c2w)
    composed = w2c[:, :3] @ c2w[:3, :3], w2c[:, :3] @ c2w[:3, 3] + w2c[:, 3]
    np.testing.assert_allclose(composed[0], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(composed[1], np.zeros(3), atol=1e-6)


def test_pose_to_w2c_anchor_recenters_world():
    # With anchor = camera position, the recentered camera sits at the world origin.
    c2w = np.eye(4)
    c2w[:3, 3] = [1e4, -2e4, 30.0]
    w2c = WaymoDataset.waymo_pose_to_w2c(c2w, anchor=c2w[:3, 3])
    np.testing.assert_allclose(w2c[:, 3], np.zeros(3), atol=1e-6)
    # A world point at the anchor maps to the camera origin.
    w2c2 = WaymoDataset.waymo_pose_to_w2c(c2w, anchor=[1e4, -2e4, 0.0])
    np.testing.assert_allclose(w2c2[:, 3], [0.0, 0.0, -30.0], atol=1e-6)


def test_pose_to_w2c_rejects_bad_pose():
    with pytest.raises(ValueError, match="non-finite"):
        WaymoDataset.waymo_pose_to_w2c(np.full((4, 4), np.inf))
    with pytest.raises(ValueError, match="4,4"):
        WaymoDataset.waymo_pose_to_w2c(np.eye(3))


def test_read_waymo_depth_synthetic_exr(tmp_path):
    import cv2

    arr = np.array([[0.0, 12.5], [-1.0, 3.25]], dtype=np.float32)
    p = str(tmp_path / "00000_1.exr")
    assert cv2.imwrite(p, arr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
    depth = WaymoDataset.read_waymo_depth(p)
    assert depth.dtype == np.float32
    # already meters (scale 1.0); 0 stays invalid; negatives defensively -> 0
    np.testing.assert_allclose(depth, [[0.0, 12.5], [0.0, 3.25]])
    with pytest.raises(FileNotFoundError):
        WaymoDataset.read_waymo_depth(str(tmp_path / "missing.exr"))


def test_waymo_intrinsics_assembly_override_and_error():
    K_raw = np.array([[549.23, 0.0, 253.61], [0.0, 549.23, 168.69], [0.0, 0.0, 1.0]])
    K = WaymoDataset.waymo_intrinsics(K_raw)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, K_raw, rtol=1e-6)
    K2 = WaymoDataset.waymo_intrinsics(K_raw, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0 and K2[1, 2] == 50.0
    with pytest.raises(ValueError, match="intrinsics"):
        WaymoDataset.waymo_intrinsics(np.eye(4))
    with pytest.raises(ValueError, match="intrinsics"):
        WaymoDataset.waymo_intrinsics(np.full((3, 3), np.nan))


def test_parse_segment_listing_groups_and_sorts():
    files = [
        "00002_1.jpg", "00000_1.jpg", "00001_1.jpg",      # cam1, out of order
        "00000_4.jpg", "00001_4.jpg",                     # cam4
        "00000_1.exr", "00000_1.npz",                     # non-jpg: ignored
        "invalid_files.h5", "readme.txt", "0000_1.jpg",   # strays: ignored
    ]
    per_cam = WaymoDataset.parse_segment_listing(files)
    assert per_cam == {1: [0, 1, 2], 4: [0, 1]}


def test_segment_short_name():
    full = "segment-123_456_with_camera_labels.tfrecord"
    assert WaymoDataset.segment_short_name(full) == "segment-123_456"
    assert WaymoDataset.segment_short_name("segment-9.tfrecord") == "segment-9"
    assert WaymoDataset.segment_short_name("plain") == "plain"


def test_filter_blacklisted_frames():
    frames = [(f"{n:05d}_1.jpg", f"{n:05d}_1.exr", f"{n:05d}_1.npz", n) for n in range(4)]
    kept = WaymoDataset.filter_blacklisted_frames(frames, 1, {"1_00001", "1_00003", "2_00000"})
    assert [fr[3] for fr in kept] == [0, 2]
    # empty/None token set is a no-op (the h5py-unavailable default)
    assert WaymoDataset.filter_blacklisted_frames(frames, 1, frozenset()) == frames
    assert WaymoDataset.filter_blacklisted_frames(frames, 1, None) == frames


# --- _frames blacklist-shortfall behavior (no data required: fake tree) ---


def _fake_waymo_tree(tmp_path, n_frames=30, cam=1):
    """Minimal on-disk layout for _frames(): only jpg NAMES are listed, never
    read, so empty files suffice."""
    seg = tmp_path / "train" / "segment-0000_with_camera_labels.tfrecord"
    seg.mkdir(parents=True)
    for n in range(n_frames):
        (seg / f"{n:05d}_{cam}.jpg").touch()
    return str(tmp_path)


def _fake_waymo_ds(root, **kwargs):
    return WaymoDataset(
        common_conf=_common_conf(),
        split="train",
        WAYMO_DIR=root,
        len_train=10,
        cameras=(1,),
        **kwargs,
    )


def test_frames_blacklist_shortfall_falls_back_to_unfiltered(tmp_path, caplog):
    """The REAL invalid_files.h5 blacklists whole-camera runs (~195/199 frames
    of one camera), which would drop a sequence below min_num_images. _frames
    must NOT raise lazily (that kills a DataLoader worker mid-training and
    breaks ComposedDataset enumeration) -- it falls back to the unfiltered
    frames with a warning."""
    root = _fake_waymo_tree(tmp_path, n_frames=30)
    ds = _fake_waymo_ds(root, use_blacklist=True, min_num_images=24)
    [seq_name] = ds.sequence_list
    short, _cam = ds.data_store[seq_name]
    # Seed the blacklist cache (bypasses h5py) with a whole-camera run:
    # 28 of 30 frames flagged -> only 2 survive filtering (< 24).
    ds._blacklist_cache[short] = frozenset(f"1_{n:05d}" for n in range(28))
    with caplog.at_level(logging.WARNING):
        frames = ds._frames(seq_name)
    assert len(frames) == 30  # blacklist ignored for this sequence, no raise
    assert any("ignoring the blacklist" in r.getMessage() for r in caplog.records)
    # and the contract surface built on _frames keeps working
    assert ds.sequence_num_frames(0) == 30


def test_frames_blacklist_filters_when_enough_frames_remain(tmp_path):
    """A mild blacklist (sequence stays >= min_num_images) is still applied."""
    root = _fake_waymo_tree(tmp_path, n_frames=30)
    ds = _fake_waymo_ds(root, use_blacklist=True, min_num_images=24)
    [seq_name] = ds.sequence_list
    short, _cam = ds.data_store[seq_name]
    ds._blacklist_cache[short] = frozenset({"1_00000", "1_00005"})
    frames = ds._frames(seq_name)
    assert len(frames) == 28
    assert all(fr[3] not in (0, 5) for fr in frames)


def test_frames_too_few_on_disk_still_raises(tmp_path):
    """A genuinely short ON-DISK listing (not blacklist-caused) still raises:
    the lazy counterpart of the eager vendors' construction-time skip. Never
    fires on the real copy (every camera has >=171 frames)."""
    root = _fake_waymo_tree(tmp_path, n_frames=10)
    ds = _fake_waymo_ds(root, min_num_images=24)
    with pytest.raises(ValueError, match="min_num_images"):
        ds._frames(ds.sequence_list[0])


# --- Waymo integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_WAYMO, reason=f"Waymo data not found at {WAYMO_DIR}")
def test_waymo_sample_schema_and_conventions():
    ds = WaymoDataset(
        common_conf=_common_conf(),
        split="train",
        WAYMO_DIR=WAYMO_DIR,
        sequences=[SEQ_CAM1],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.CAMERA_ID in ds.available_modalities
    assert Modality.SKY_MASK not in ds.available_modalities   # sky not encoded in depth
    assert Modality.TIMESTAMP not in ds.available_modalities  # nothing on disk

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
    # recentered per segment: world-frame translations are float32-safe, not ~1e4 m
    assert np.abs(extr[:, :, 3]).max() < 1000.0
    assert (depth > 0).any()                              # sparse LiDAR: some valid depth
    assert (depth >= 0).all()                             # 0=invalid, no negatives (no sky)
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["camera_ids"].dtype == np.int32
    assert (batch["camera_ids"] == 1).all()               # single-camera sequence
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_WAYMO, reason=f"Waymo data not found at {WAYMO_DIR}")
def test_waymo_reprojection_closure():
    """Frame A's valid world points reprojected into frame B land at depths
    matching B's sparse LiDAR depth (locks depth-scale x pose-convention x
    intrinsics consistency end-to-end through process_one_image)."""
    ds = WaymoDataset(
        common_conf=_eval_common(),
        split="train",
        WAYMO_DIR=WAYMO_DIR,
        sequences=[SEQ_CAM1],
        len_train=10,
    )
    h, w = ds.native_image_size(0)
    b = ds.get_data(seq_name=ds.sequence_list[0], ids=np.array([0, 3]), aspect_ratio=h / w)
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
    assert np.median(rel_err) < 0.05


@pytest.mark.skipif(not HAVE_WAYMO, reason=f"Waymo data not found at {WAYMO_DIR}")
def test_waymo_getitem_tuple_index():
    ds = WaymoDataset(
        common_conf=_common_conf(),
        split="train",
        WAYMO_DIR=WAYMO_DIR,
        sequences=[SEQ_CAM1],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_WAYMO, reason=f"Waymo data not found at {WAYMO_DIR}")
def test_waymo_side_camera_mixed_resolution():
    """Cams 4/5 have a different native resolution (236x512 vs 341x512); the
    loader must handle per-camera shapes and report the right native size."""
    ds = WaymoDataset(
        common_conf=_eval_common(),
        split="train",
        WAYMO_DIR=WAYMO_DIR,
        sequences=[SEQ_CAM1, SEQ_CAM4],
        len_train=10,
    )
    i1 = ds.sequence_list.index(SEQ_CAM1)
    i4 = ds.sequence_list.index(SEQ_CAM4)
    assert ds.native_image_size(i1) == (341, 512)
    assert ds.native_image_size(i4) == (236, 512)
    b = ds.get_data(seq_name=SEQ_CAM4, ids=np.array([0, 5]), aspect_ratio=236 / 512)
    assert (np.stack(b["camera_ids"]) == 4).all()
    img = np.stack(b["images"])
    assert img.shape[1:3] == (224, 512)  # 512 * (236/512) = 236 -> /16 snap = 224
    assert (np.stack(b["depths"]) > 0).any()


@pytest.mark.skipif(not HAVE_WAYMO, reason=f"Waymo data not found at {WAYMO_DIR}")
def test_waymo_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _waymo_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "camera_ids" in sample                  # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_WAYMO, reason=f"Waymo data not found at {WAYMO_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The
    two must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _waymo_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 9, 3, 14]   # deliberately unordered
    sample = composed.get_sample(0, ids=ids, aspect_ratio=0.75)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample
    np.testing.assert_array_equal(sample["ids"].numpy(), ids)

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

    # Order honored verbatim: per-frame extrinsics (deterministic in eval mode,
    # ego moves ~0.6 m/frame so frames are distinguishable) must match a sorted-ids
    # load frame-for-frame under the requested permutation.
    batch_sorted = vendor.get_data(
        seq_name=composed.sequence_name(0), ids=np.array(sorted(ids)), aspect_ratio=0.75
    )
    sorted_ids = sorted(ids)
    for k, fid in enumerate(ids):
        np.testing.assert_array_equal(
            batch["extrinsics"][k], batch_sorted["extrinsics"][sorted_ids.index(fid)]
        )
    t = np.stack(batch["extrinsics"])[:, :, 3]
    assert np.linalg.norm(t[0] - t[1]) > 0.05   # distinct frames really differ


@pytest.mark.skipif(not HAVE_WAYMO, reason=f"Waymo data not found at {WAYMO_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate
    sequences. One segment pattern expands to its 5 single-camera sequences."""
    from hydra.utils import instantiate

    composed = instantiate(
        _waymo_dataset_cfg(seqs=[SEGMENT]), common_config=_eval_common(), _recursive_=False
    )
    vendor = composed.base_dataset.datasets[0]
    assert composed.num_sequences() == 5                  # one segment x 5 cameras
    assert set(vendor.sequence_list) == {f"{SEGMENT}/cam{c}" for c in (1, 2, 3, 4, 5)}
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in vendor.sequence_list
        n = composed.sequence_num_frames(gi)
        assert n == len(vendor._frames(name))
        assert n > 100                                    # survey: ~196-199 frames/camera


@pytest.mark.skipif(not HAVE_WAYMO, reason=f"Waymo data not found at {WAYMO_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _waymo_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (341, 512)                       # Waymo front cam native (H, W)

    composed.set_img_size(512)                        # native long side
    assert composed.img_size == 512
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (336, 512)   # 341 -> /16 snapped

    composed.set_img_size(256)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (160, 256)
