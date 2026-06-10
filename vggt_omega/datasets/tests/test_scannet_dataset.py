import os
import time

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.scannet import ScannetDataset

SCANNET_DIR = "/jfs/Data_4DFF/train_data/scannet"
HAVE_SCANNET = os.path.isdir(os.path.join(SCANNET_DIR, "scans_train"))
# scene0526_00 is one of the smallest scenes (340 frames); use it for fast tests.
SCENE = "scene0526_00"
SCENE2 = "scene0100_00"


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


def _scannet_dataset_cfg(seqs=(SCENE,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.scannet.ScannetDataset",
                "split": "train",
                "SCANNET_DIR": SCANNET_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- ScanNet-specific helper unit tests (no data required) ---


def test_scannet_pose_to_w2c_inverts_c2w():
    # camera-to-world = pure translation by (1,2,3), identity rotation
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = ScannetDataset.scannet_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_scannet_pose_to_w2c_axis_remap_roundtrip():
    # camera-to-world = 90 deg rotation about z plus translation; a point at
    # (0,0,1) in the camera frame must map world->camera back onto itself.
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [1.0, 0.0, 0.0]
    p_cam = np.array([0.0, 0.0, 1.0])
    p_world = rot @ p_cam + c2w[:3, 3]
    w2c = ScannetDataset.scannet_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:, :3] @ p_world + w2c[:, 3], p_cam, atol=1e-6)
    np.testing.assert_allclose(w2c[:, :3], rot.T, atol=1e-6)


def test_scannet_pose_to_w2c_rejects_non_finite_and_bad_shape():
    bad = np.eye(4)
    bad[1, 3] = -np.inf  # raw-ScanNet tracking-failure encoding
    with pytest.raises(ValueError, match="non-finite"):
        ScannetDataset.scannet_pose_to_w2c(bad)
    with pytest.raises(ValueError, match="4,4"):
        ScannetDataset.scannet_pose_to_w2c(np.eye(3))


def test_read_scannet_cam_roundtrip_and_missing_key(tmp_path):
    K = np.array([[577.6, 0.0, 318.9], [0.0, 578.7, 242.7], [0.0, 0.0, 1.0]], np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [0.5, -0.25, 1.5]
    good = tmp_path / "00000.npz"
    np.savez(good, intrinsics=K, pose=c2w)
    K2, c2w2 = ScannetDataset.read_scannet_cam(str(good))
    assert K2.dtype == np.float32 and c2w2.dtype == np.float64
    np.testing.assert_allclose(K2, K)
    np.testing.assert_allclose(c2w2, c2w)

    bad = tmp_path / "00001.npz"
    np.savez(bad, intrinsics=K)  # no 'pose'
    with pytest.raises(ValueError, match="pose"):
        ScannetDataset.read_scannet_cam(str(bad))


def test_scannet_depth_reader_units_and_invalid(tmp_path):
    import cv2

    arr = np.array([[0, 1000], [2000, 500]], dtype=np.uint16)
    p = tmp_path / "00000.png"
    assert cv2.imwrite(str(p), arr)
    depth = ScannetDataset.read_scannet_depth(str(p))
    np.testing.assert_allclose(depth, [[0.0, 1.0], [2.0, 0.5]])  # mm->m, 0 invalid
    assert depth.dtype == np.float32
    with pytest.raises(FileNotFoundError, match="depth"):
        ScannetDataset.read_scannet_depth(str(tmp_path / "missing.png"))


def test_scannet_intrinsics_passthrough_override_and_error():
    K = np.array([[577.6, 0.0, 318.9], [0.0, 578.7, 242.7], [0.0, 0.0, 1.0]])
    out = ScannetDataset.scannet_intrinsics(K)
    assert out.shape == (3, 3) and out.dtype == np.float32 and out[2, 2] == 1.0
    np.testing.assert_allclose(out, K)
    K2 = ScannetDataset.scannet_intrinsics(K, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0
    with pytest.raises(ValueError, match="intrinsics"):
        ScannetDataset.scannet_intrinsics(None)
    with pytest.raises(ValueError, match="invalid K"):
        ScannetDataset.scannet_intrinsics(np.eye(4))
    bad = K.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid K"):
        ScannetDataset.scannet_intrinsics(bad)


def test_scannet_constructor_default_depth_scale_through_get_data(tmp_path):
    """Lock the CONSTRUCTOR default depth_scale (=1000, mm->m) end-to-end through
    get_data, not just read_scannet_depth's own default. A synthetic frame with
    constant 1500 mm depth must come back as exactly 1.5 m when the vendor is
    built WITHOUT an explicit depth_scale -- a regression of the __init__ default
    (e.g. a copy-paste of TUM's 5000.0) would return 0.3 m and fail here. The
    cross-frame reprojection closure test canNOT catch this: a uniform depth
    scaling nearly cancels in the closure when the baseline is small vs depth."""
    import cv2

    scene = tmp_path / "scans_train" / "scene9999_00"
    for sub in ("color", "depth", "cam"):
        (scene / sub).mkdir(parents=True)
    rng = np.random.default_rng(0)
    assert cv2.imwrite(
        str(scene / "color" / "00000.jpg"),
        rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
    )
    assert cv2.imwrite(
        str(scene / "depth" / "00000.png"),
        np.full((480, 640), 1500, dtype=np.uint16),  # constant 1.5 m in mm
    )
    K = np.array([[577.6, 0.0, 318.9], [0.0, 578.7, 242.7], [0.0, 0.0, 1.0]], np.float32)
    np.savez(scene / "cam" / "00000.npz", intrinsics=K, pose=np.eye(4, dtype=np.float32))

    conf = OmegaConf.merge(
        _common_conf(),
        OmegaConf.create(
            {
                "training": False,           # disable random scale aug
                "rescale_aug": False,        # deterministic resize
                "get_nearby": False,         # honor ids verbatim
                "allow_duplicate_img": False,
                "augs": {"scales": None},
            }
        ),
    )
    ds = ScannetDataset(
        common_conf=conf,
        split="train",
        SCANNET_DIR=str(tmp_path),
        len_train=10,
        min_num_images=1,
        # NB: depth_scale deliberately NOT passed -- this exercises the default.
    )
    batch = ds.get_data(seq_name="scene9999_00", ids=np.array([0]), aspect_ratio=0.75)
    depth = batch["depths"][0]
    assert depth.size > 0
    np.testing.assert_allclose(depth, 1.5, rtol=1e-6)  # mm -> m via the default


def test_scannet_min_num_images_checked_lazily(tmp_path):
    """A too-short scene must construct fine (lazy enumeration) but raise a
    clear ValueError at first per-frame access."""
    scene = tmp_path / "scans_train" / "scene9999_00"
    for sub in ("color", "depth", "cam"):
        (scene / sub).mkdir(parents=True)
    for i in range(3):
        (scene / "color" / f"{i:05d}.jpg").touch()
    ds = ScannetDataset(
        common_conf=_common_conf(), split="train", SCANNET_DIR=str(tmp_path), len_train=10
    )
    assert ds.sequence_list == ["scene9999_00"]
    assert ds.data_store == {}  # nothing enumerated at construction
    with pytest.raises(ValueError, match="min_num_images"):
        ds.sequence_num_frames(0)


# --- ScanNet integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_scannet_lazy_construction_over_all_scenes():
    """Construction over the full 1510-scene root must stay cheap (one dir
    listing; no per-frame enumeration) -- the network-FS scalability contract."""
    t0 = time.monotonic()
    ds = ScannetDataset(
        common_conf=_common_conf(), split="train", SCANNET_DIR=SCANNET_DIR, len_train=10
    )
    elapsed = time.monotonic() - t0
    assert ds.sequence_list_len >= 1000
    assert ds.data_store == {}  # frame lists deferred to first access
    assert elapsed < 10.0, f"construction took {elapsed:.1f}s; should be a single listing"


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_scannet_sample_schema_and_conventions():
    ds = ScannetDataset(
        common_conf=_common_conf(),
        split="train",
        SCANNET_DIR=SCANNET_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities

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
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert (depth >= 0).all()                             # indoor: no sky encoding
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["seq_name"] == "scannet_" + SCENE
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)

    # out-of-range index gives the documented clear error, not IndexError
    with pytest.raises(ValueError, match="out of range"):
        ds.get_data(seq_index=999, img_per_seq=2, aspect_ratio=1.0)


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_scannet_cross_frame_reprojection_closure():
    """Frame A's valid world points reprojected into frame B must land at depths
    matching B's depth map (locks pose-convention x intrinsics consistency
    end-to-end through process_one_image; survey gold check). NB: a UNIFORM
    depth-scale error nearly cancels in this closure (small baseline vs scene
    depth), so the depth scale is locked separately: by the synthetic
    constructor-default test above and by the absolute-magnitude assertion below.

    The [0, 30] frame gap and the 0.03 threshold are chosen from measurement:
    correct convention closes at median rel err ~0.005, while skipping the
    w2c inversion (passing c2w through) gives ~0.18 -- 6x headroom both ways,
    vs only ~6% margin at the low-motion [0, 10] pair with a 0.05 threshold."""
    ds = ScannetDataset(
        common_conf=_eval_common(),
        split="train",
        SCANNET_DIR=SCANNET_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    b = ds.get_data(seq_name=ds.sequence_list[0], ids=np.array([0, 30]), aspect_ratio=0.75)
    world = np.stack(b["world_points"])
    pmask = np.stack(b["point_masks"])
    extr = np.stack(b["extrinsics"])
    intr = np.stack(b["intrinsics"])
    depth = np.stack(b["depths"])

    # Absolute metric magnitude (survey: indoor, valid p50 ~1.3-2.0 m). A 2x
    # depth-scale error (e.g. /2000) gives p50 ~0.8 m and fails this even
    # though the closure below would still pass.
    d_valid = depth[depth > 0]
    assert d_valid.size > 0
    assert 1.0 < np.median(d_valid) < 2.5

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


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_scannet_getitem_tuple_index():
    ds = ScannetDataset(
        common_conf=_common_conf(),
        split="train",
        SCANNET_DIR=SCANNET_DIR,
        sequences=[SCENE],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_scannet_frame_step_subsamples():
    full = ScannetDataset(
        common_conf=_common_conf(), split="train",
        SCANNET_DIR=SCANNET_DIR, sequences=[SCENE], len_train=10,
    )
    stepped = ScannetDataset(
        common_conf=_common_conf(), split="train",
        SCANNET_DIR=SCANNET_DIR, sequences=[SCENE], len_train=10, frame_step=10,
    )
    n_full = full.sequence_num_frames(0)
    n_step = stepped.sequence_num_frames(0)
    assert n_step == (n_full + 9) // 10


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_scannet_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _scannet_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" in sample                   # extra registered key carried through
    assert "timestamps" in sample                  # synthesized at the nominal 30 fps
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The
    two must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _scannet_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 12, 4, 30]                                    # deliberately unordered
    sample = composed.get_sample(0, ids=ids, aspect_ratio=0.75)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample
    np.testing.assert_array_equal(sample["ids"].numpy(), ids)

    # Drift guard: the same vendor.get_data + manual tensorize must match byte-for-byte.
    vendor = composed.base_dataset.datasets[0]
    seq_name = composed.sequence_name(0)
    batch = vendor.get_data(seq_name=seq_name, ids=np.array(ids), aspect_ratio=0.75)
    manual = (
        torch.from_numpy(np.stack(batch["images"]).astype(np.float32))
        .permute(0, 3, 1, 2)
        .to(torch.get_default_dtype())
        .div(255)
    )
    torch.testing.assert_close(sample["images"], manual)

    # Order honored verbatim: in deterministic eval mode each returned frame must
    # equal an independent single-id fetch of the SAME frame id (per-frame depth
    # maps differ between frames, so a sort/shuffle would be caught).
    for k, fid in enumerate(ids):
        single = vendor.get_data(seq_name=seq_name, ids=np.array([fid]), aspect_ratio=0.75)
        np.testing.assert_allclose(sample["depths"][k].numpy(), single["depths"][0])
        np.testing.assert_allclose(
            sample["extrinsics"][k].numpy(), single["extrinsics"][0], atol=1e-6
        )


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendor's real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SCENE, SCENE2]
    composed = instantiate(
        _scannet_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        n = composed.sequence_num_frames(gi)
        assert n >= vendor.min_num_images
        # the lazy load triggered above must agree with the cached frame list
        assert n == len(vendor.data_store[name])


@pytest.mark.skipif(not HAVE_SCANNET, reason=f"ScanNet data not found at {SCANNET_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _scannet_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (480, 640)                       # preprocessed ScanNet VGA (H, W)

    composed.set_img_size(640)                        # native long side
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
