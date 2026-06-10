import logging
import os
import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.dl3dv import Dl3dvDataset

DL3DV_DIR = "/jfs/Data_4DFF/train_data/dl3dv"
HAVE_DL3DV = os.path.isdir(DL3DV_DIR)
# Two known scenes (DL3DV scene dirs are 64-char hex hashes); restrict
# `sequences` to these so integration tests never touch the other ~6k scenes.
SEQ_A = "0003dc82473fd52c53dcbdc2deb4e6e9c3548d6f8c9b03f9ea8d3c7d3ce33546"
SEQ_B = "0010315801d030548e39764dd62b1cb59a43650f93e8ecd001ffca31aff91c44"
# A real DL3DV scene with only 12 frames (< the default min_num_images=24).
# Short scenes EXIST in this export -- an exhaustive count found 7 of 6378
# scenes with 12-22 frames -- so the vendor must drop them at construction;
# a lazy ValueError would crash random training draws and deterministic
# full-root inference enumeration.
SEQ_SHORT = "2e00e376bd0e32e6f10003c5a498d4e05ad3923ad7cbfdf601323ada6bcfe9cb"


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


def _dl3dv_dataset_cfg(seqs=(SEQ_A,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.dl3dv.Dl3dvDataset",
                "split": "train",
                "DL3DV_DIR": DL3DV_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


def _rot_z(deg):
    a = np.deg2rad(deg)
    return np.array(
        [
            [np.cos(a), -np.sin(a), 0.0],
            [np.sin(a), np.cos(a), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


# --- DL3DV-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    w2c = Dl3dvDataset.dl3dv_pose_to_w2c(np.eye(4))
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-6)

    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = Dl3dvDataset.dl3dv_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_rotation_and_optical_axis():
    """c2w with rotation+translation: w2c must be the exact rigid inverse, and a
    world point 1 unit along the camera's optical axis (c2w z-column) must land
    at (0,0,1) in the camera frame -- locks the OpenCV axis convention."""
    R = _rot_z(90.0)
    t = np.array([1.0, 2.0, 3.0])
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = Dl3dvDataset.dl3dv_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], R.T, atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], -R.T @ t, atol=1e-6)
    p_world = t + R[:, 2]  # 1 unit along the optical (z-forward) axis
    p_cam = w2c[:3, :3] @ p_world + w2c[:, 3]
    np.testing.assert_allclose(p_cam, [0.0, 0.0, 1.0], atol=1e-6)


def test_pose_to_w2c_rejects_bad_input():
    with pytest.raises(ValueError, match="shape"):
        Dl3dvDataset.dl3dv_pose_to_w2c(np.eye(3))
    bad = np.eye(4)
    bad[0, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        Dl3dvDataset.dl3dv_pose_to_w2c(bad)


def test_intrinsics_assembly_override_and_error():
    raw = np.array([[437.0, 0.0, 482.5], [0.0, 437.2, 266.5], [0.0, 0.0, 1.0]])
    K = Dl3dvDataset.dl3dv_intrinsics(raw)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, raw, atol=1e-4)
    K2 = Dl3dvDataset.dl3dv_intrinsics(raw, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0
    with pytest.raises(ValueError, match="intrinsic"):
        Dl3dvDataset.dl3dv_intrinsics(None)
    with pytest.raises(ValueError, match="shape"):
        Dl3dvDataset.dl3dv_intrinsics(np.zeros(4))


def test_read_camera_npz_and_missing_key(tmp_path):
    R = _rot_z(30.0)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = [0.5, -1.0, 2.0]
    K = np.array([[440.0, 0.0, 480.0], [0.0, 441.0, 270.0], [0.0, 0.0, 1.0]])
    p = tmp_path / "frame_00001.npz"
    np.savez(p, pose=c2w, intrinsic=K)
    w2c, K_out = Dl3dvDataset.read_dl3dv_camera(str(p))
    np.testing.assert_allclose(w2c, Dl3dvDataset.dl3dv_pose_to_w2c(c2w), atol=1e-7)
    np.testing.assert_allclose(K_out, K, atol=1e-4)
    assert w2c.dtype == np.float32 and K_out.dtype == np.float32

    p_bad = tmp_path / "frame_00002.npz"
    np.savez(p_bad, intrinsic=K)  # no 'pose'
    with pytest.raises(ValueError, match="pose"):
        Dl3dvDataset.read_dl3dv_camera(str(p_bad))


def test_empty_depth_is_all_invalid():
    d = Dl3dvDataset.empty_depth(4, 6)
    assert d.shape == (4, 6) and d.dtype == np.float32
    assert (d == 0).all()           # 0 = invalid everywhere (no depth modality)
    assert not (d < 0).any()        # and never sky


def _make_fake_scene(root, name, n_frames):
    """Synthetic DL3DV scene layout: construction-time counting only reads dir
    entry NAMES (never pixels), so empty touch()ed files are sufficient."""
    rgb = root / name / "dense" / "rgb"
    cam = root / name / "dense" / "cam"
    rgb.mkdir(parents=True)
    cam.mkdir(parents=True)
    for i in range(1, n_frames + 1):
        (rgb / f"frame_{i:05d}.png").touch()
        (cam / f"frame_{i:05d}.npz").touch()


def test_short_scenes_dropped_at_construction(tmp_path, caplog):
    """Scenes with < min_num_images frames must be filtered out (with a warning)
    when the dataset is built -- the TUM/7-Scenes contract -- NOT raise lazily on
    first access, which would crash DataLoader workers on the 7 real short scenes
    (12-22 frames) in the full root. Synthetic root, no /jfs needed."""
    long_name, short_name, broken_name = "a" * 64, "b" * 64, "c" * 64
    _make_fake_scene(tmp_path, long_name, 30)
    _make_fake_scene(tmp_path, short_name, 5)
    (tmp_path / broken_name).mkdir()          # no dense/rgb at all -> count 0

    with caplog.at_level(logging.WARNING):
        ds = Dl3dvDataset(
            common_conf=_common_conf(),
            split="train",
            DL3DV_DIR=str(tmp_path),
            len_train=10,
            min_num_images=24,
        )
    # Only the long scene survives; sequence_list is fixed and index-stable.
    assert ds.sequence_list == [long_name]
    assert ds.sequence_list_len == 1
    assert any("only 5 frames" in m for m in caplog.messages)   # warned, not raised
    assert any("only 0 frames" in m for m in caplog.messages)
    # Frame count is served from the construction count WITHOUT loading the
    # lazy frame-path list; the dropped scenes are simply unknown afterwards.
    assert ds.sequence_num_frames(0) == 30
    assert ds.data_store[long_name] is None
    with pytest.raises(KeyError, match="unknown sequence"):
        ds._frames(short_name)
    # The lazy listing still works on demand for the kept scene.
    assert len(ds._frames(long_name)) == 30
    assert ds.data_store[long_name] is not None

    # If filtering leaves nothing usable, construction fails loudly.
    with pytest.raises(ValueError, match="No usable DL3DV sequences"):
        Dl3dvDataset(
            common_conf=_common_conf(),
            split="train",
            DL3DV_DIR=str(tmp_path),
            sequences=[short_name],
            len_train=10,
            min_num_images=24,
        )


# --- DL3DV integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_dl3dv_sample_schema_and_conventions():
    ds = Dl3dvDataset(
        common_conf=_common_conf(),
        split="train",
        DL3DV_DIR=DL3DV_DIR,
        sequences=[SEQ_A],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.DEPTH not in ds.available_modalities       # no depth on disk
    assert Modality.WORLD_POINTS not in ds.available_modalities

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    img = np.stack(batch["images"])
    depth = np.stack(batch["depths"])
    extr = np.stack(batch["extrinsics"])
    intr = np.stack(batch["intrinsics"])
    world = np.stack(batch["world_points"])
    cam = np.stack(batch["cam_points"])
    pmask = np.stack(batch["point_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    # No depth modality: zero depth, no valid points, no sky key, zeroed geometry.
    assert (depth == 0).all()
    assert not pmask.any()
    assert "sky_masks" not in batch                       # unadvertised -> key not emitted
    assert (world == 0).all() and (cam == 0).all()
    assert batch["seq_name"] == "dl3dv_" + SEQ_A
    assert batch["is_metric"] is False and batch["is_video"] is True
    assert "timestamps" not in batch                      # TIMESTAMP not advertised

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_dl3dv_pose_and_intrinsics_sanity():
    """Substitute for the depth reprojection-closure test (DL3DV has no depth):
    returned extrinsics must be proper rigid world->cam transforms (orthonormal
    R, det=+1), camera centers must move smoothly along the video, and K must be
    plausible against the processed image size (pp centered by process_one_image,
    sane focal). Raw npz intrinsics must have pp exactly at the native center."""
    ds = Dl3dvDataset(
        common_conf=_eval_common(),
        split="train",
        DL3DV_DIR=DL3DV_DIR,
        sequences=[SEQ_A],
        len_train=10,
    )
    ids = [0, 5, 10, 40]
    b = ds.get_data(seq_name=SEQ_A, ids=np.array(ids), aspect_ratio=0.75)
    extr = np.stack(b["extrinsics"])
    intr = np.stack(b["intrinsics"])
    H, W = np.stack(b["images"]).shape[1:3]

    centers = []
    for E in extr:
        R, t = E[:3, :3], E[:, 3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)   # orthonormal
        assert np.linalg.det(R) > 0.99                              # proper rotation
        centers.append(-R.T @ t)
    centers = np.stack(centers)
    assert np.isfinite(centers).all()
    # Frames are an ordered video: the camera moves, and nearby frames are
    # closer than far-apart ones (0->5 vs 0->40).
    d05 = np.linalg.norm(centers[1] - centers[0])
    d040 = np.linalg.norm(centers[3] - centers[0])
    assert d05 > 1e-6 and d040 > d05

    for K in intr:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        assert fx > 0 and fy > 0
        assert 0.85 < fx / fy < 1.15                       # near-isotropic focal
        assert 0.2 * W < fx < 5.0 * W                      # plausible FoV
        assert abs(cx - W / 2) <= 2 and abs(cy - H / 2) <= 2  # pp centered by crop

    # Raw (unprocessed) intrinsics: principal point exactly at the native center.
    h, w = ds.native_image_size(0)
    w2c, K_raw = Dl3dvDataset.read_dl3dv_camera(ds.data_store[SEQ_A][0][1])
    assert abs(K_raw[0, 2] - w / 2) < 1e-3
    assert abs(K_raw[1, 2] - h / 2) < 1e-3


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_dl3dv_getitem_tuple_index():
    ds = Dl3dvDataset(
        common_conf=_common_conf(),
        split="train",
        DL3DV_DIR=DL3DV_DIR,
        sequences=[SEQ_A],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_dl3dv_known_short_scene_excluded_but_long_scene_sampleable():
    """Default-config construction over a mix of a long and a REAL 12-frame scene
    must keep only the long one, and get_data must keep working -- before the
    construction-time filter this exact setup raised ValueError from _frames and
    killed the DataLoader worker."""
    ds = Dl3dvDataset(
        common_conf=_common_conf(),
        split="train",
        DL3DV_DIR=DL3DV_DIR,
        sequences=[SEQ_A, SEQ_SHORT],
        len_train=10,
    )
    assert ds.sequence_list == [SEQ_A]            # short scene dropped up front
    assert SEQ_SHORT not in ds._frame_counts
    batch = ds.get_data(seq_index=0, img_per_seq=2, aspect_ratio=1.0)
    assert batch["seq_name"] == "dl3dv_" + SEQ_A
    assert batch["frame_num"] == 2
    with pytest.raises(KeyError, match="unknown sequence"):
        ds._frames(SEQ_SHORT)


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_dl3dv_full_root_construction_filters_short_scenes_and_stays_lazy():
    """Construction over ALL ~6378 scenes does a root scandir plus a threaded
    per-scene frame COUNT (names only; ~12-18s on this network FS) to drop the
    7 real short scenes (12-22 frames) up front; full frame-path lists stay
    unloaded until used. Every surviving index must then be enumerable AND
    sampleable -- the exact full-root default training/inference configuration
    that used to crash."""
    t0 = time.monotonic()
    ds = Dl3dvDataset(
        common_conf=_common_conf(), split="train", DL3DV_DIR=DL3DV_DIR, len_train=10
    )
    elapsed = time.monotonic() - t0
    assert ds.sequence_list_len > 6000
    assert elapsed < 60.0, f"construction took {elapsed:.1f}s (count not threaded?)"
    assert all(v is None for v in ds.data_store.values())   # frame paths still lazy
    # The known 12-frame scene exists on disk but must NOT be in the dataset.
    assert os.path.isdir(os.path.join(DL3DV_DIR, SEQ_SHORT))
    assert SEQ_SHORT not in ds.sequence_list
    # Exhaustive guard over EVERY kept scene (this is the deterministic
    # full-root inference enumeration that previously raised on short scenes):
    # counts cover exactly sequence_list and all clear min_num_images.
    assert set(ds._frame_counts) == set(ds.sequence_list)
    assert all(
        ds.sequence_num_frames(i) >= ds.min_num_images
        for i in range(ds.sequence_list_len)
    )
    # Enumeration above was served from construction counts (still lazy)...
    assert all(v is None for v in ds.data_store.values())
    # ...and the lazy frame-path listing loads on demand, consistent with the
    # construction-time count.
    name = ds.sequence_list[0]
    assert len(ds._frames(name)) == ds._frame_counts[name]
    assert ds.data_store[name] is not None


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_dl3dv_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _dl3dv_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert sample["extrinsics"].shape == (4, 3, 4)
    assert "timestamps" not in sample              # not advertised, not fabricated
    assert "modalities" in sample
    assert sample["modalities"] == sorted(
        m.value for m in Dl3dvDataset.AVAILABLE
    )                                              # ["extrinsics","images","intrinsics"]
    # Unadvertised non-core keys (sky_masks) are not emitted, so not carried either.
    assert "sky_masks" not in sample
    assert not bool(sample["point_masks"].any())   # zero depth -> no valid points


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested (unordered) id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _dl3dv_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]                            # deliberately NOT sorted
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

    # Order honored verbatim: frame k's returned extrinsics must equal the w2c
    # read straight from the npz of ids[k] (eval mode + landscape_check=False
    # leave extrinsics untouched by process_one_image).
    frames = vendor.data_store[composed.sequence_name(0)]
    for k, i in enumerate(ids):
        w2c, _ = Dl3dvDataset.read_dl3dv_camera(frames[i][1])
        np.testing.assert_allclose(sample["extrinsics"][k].numpy(), w2c, atol=1e-6)


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendor's real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ_A, SEQ_B]
    composed = instantiate(
        _dl3dv_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        n = composed.sequence_num_frames(gi)
        assert n == vendor.sequence_num_frames(vendor.sequence_list.index(name))
        assert n >= vendor.min_num_images
        # Enumeration is served from the construction-time count; the lazy
        # frame-path listing (loaded here on demand) must agree with it.
        assert n == len(vendor._frames(name))
        assert n == len(vendor.data_store[name])   # lazy list now cached


@pytest.mark.skipif(not HAVE_DL3DV, reason=f"DL3DV data not found at {DL3DV_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants. DL3DV resolution varies
    per scene, so the expected shapes are computed from the data, never assumed."""
    from PIL import Image
    from hydra.utils import instantiate

    composed = instantiate(
        _dl3dv_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    vendor = composed.base_dataset.datasets[0]
    with Image.open(vendor.data_store[SEQ_A][0][0]) as im:
        assert (h, w) == (im.size[1], im.size[0])  # matches the actual first frame
    assert 500 <= h <= 560 and 930 <= w <= 990     # surveyed DL3DV range

    def expected_shape(long_side):
        short = int(long_side * (h / w))
        short -= short % 16                        # patch-size snapping
        return (short, long_side)

    composed.set_img_size(640)
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == expected_shape(640)

    composed.set_img_size(320)
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == expected_shape(320)
