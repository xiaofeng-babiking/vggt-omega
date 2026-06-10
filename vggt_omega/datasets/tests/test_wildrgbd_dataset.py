import json
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.wildrgbd import WildRgbdDataset

WILDRGBD_DIR = "/jfs/Data_4DFF/train_data/wildrgbd"
HAVE_WRGBD = os.path.isdir(WILDRGBD_DIR)
# Two small train-split apple scenes (100 frames each); restrict `sequences` to
# them so integration-test construction stays fast.
SEQ = "apple/scene_002"
SEQ2 = "apple/scene_036"


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


def _wildrgbd_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.wildrgbd.WildRgbdDataset",
                "split": "train",
                "WILDRGBD_DIR": WILDRGBD_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- WildRGB-D-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = WildRgbdDataset.wildrgbd_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_rotation_inverts_c2w():
    # camera-to-world = 90-degree rotation about z + translation; w2c must be the
    # exact rigid inverse (composition with c2w gives identity).
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [1.0, 0.0, -2.0]
    w2c = WildRgbdDataset.wildrgbd_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], rot.T, atol=1e-6)
    w2c_h = np.vstack([w2c, [0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(w2c_h @ c2w, np.eye(4), atol=1e-6)


def test_pose_to_w2c_rejects_bad_input():
    with pytest.raises(ValueError, match=r"\(4,4\)"):
        WildRgbdDataset.wildrgbd_pose_to_w2c(np.eye(3))
    bad = np.eye(4)
    bad[0, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        WildRgbdDataset.wildrgbd_pose_to_w2c(bad)


def test_depth_reader_units_and_invalid(tmp_path):
    from PIL import Image

    arr = np.array([[0, 500], [1000, 4582]], dtype=np.uint16)
    p = tmp_path / "00000.png"
    Image.fromarray(arr).save(p)
    depth = WildRgbdDataset.read_wildrgbd_depth(str(p))
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[0.0, 0.5], [1.0, 4.582]])  # mm -> m, 0 invalid


def test_intrinsics_assembly_override_and_error():
    K_raw = np.array([[483.5, 0.0, 193.0], [0.0, 483.5, 256.0], [0.0, 0.0, 1.0]])
    K = WildRgbdDataset.wildrgbd_intrinsics(K_raw)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, K_raw, atol=1e-4)
    K2 = WildRgbdDataset.wildrgbd_intrinsics(K_raw, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0
    with pytest.raises(ValueError, match="intrinsics"):
        WildRgbdDataset.wildrgbd_intrinsics(None)
    with pytest.raises(ValueError, match=r"\(3,3\)"):
        WildRgbdDataset.wildrgbd_intrinsics(np.eye(4))
    bad = K_raw.copy()
    bad[0, 0] = 0.0
    with pytest.raises(ValueError, match="focal"):
        WildRgbdDataset.wildrgbd_intrinsics(bad)


def test_read_metadata_npz_roundtrip(tmp_path):
    K = np.array([[480.0, 0.0, 192.0], [0.0, 480.0, 256.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, 3] = [0.1, -0.2, 0.3]
    p = tmp_path / "00000.npz"
    np.savez(p, camera_intrinsics=K, camera_pose=c2w)
    K_out, w2c = WildRgbdDataset.read_wildrgbd_metadata(str(p))
    assert K_out.dtype == np.float32 and w2c.dtype == np.float32
    np.testing.assert_allclose(K_out, K, atol=1e-5)
    np.testing.assert_allclose(w2c[:, 3], [-0.1, 0.2, -0.3], atol=1e-6)


def test_load_split_index_parses_sorts_and_merges(tmp_path):
    train = {"apple": {"scenes/scene_002": [3, 0, 7]}}
    test = {"apple": {"scenes/scene_006": [4, 1]}}
    (tmp_path / "selected_seqs_train.json").write_text(json.dumps(train))
    (tmp_path / "selected_seqs_test.json").write_text(json.dumps(test))
    idx = WildRgbdDataset.load_split_index(str(tmp_path), "train")
    assert idx == {"apple/scene_002": [0, 3, 7]}        # ids sorted numerically
    idx_all = WildRgbdDataset.load_split_index(str(tmp_path), "all")
    assert set(idx_all) == {"apple/scene_002", "apple/scene_006"}
    with pytest.raises(ValueError, match="split"):
        WildRgbdDataset.load_split_index(str(tmp_path), "validation")


def test_sequence_pattern_matching():
    assert WildRgbdDataset._matches("apple/scene_002", ["apple"])           # category
    assert WildRgbdDataset._matches("apple/scene_002", ["scene_002"])       # scene
    assert WildRgbdDataset._matches("apple/scene_002", ["apple/scene_*"])   # full glob
    assert not WildRgbdDataset._matches("apple/scene_002", ["banana", "scene_999"])


# --- WildRGB-D integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_WRGBD, reason=f"WildRGB-D data not found at {WILDRGBD_DIR}")
def test_wildrgbd_sample_schema_and_conventions():
    ds = WildRgbdDataset(
        common_conf=_common_conf(),
        split="train",
        WILDRGBD_DIR=WILDRGBD_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    # No timestamps in WildRGB-D: must not be advertised. SKY_MASK IS advertised
    # (all-False indoors), matching the TUM/7-Scenes indoor-vendor convention.
    assert Modality.TIMESTAMP not in ds.available_modalities
    assert Modality.SKY_MASK in ds.available_modalities

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    assert batch["seq_name"] == "wildrgbd_" + SEQ
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
    assert (depth >= 0).all()                             # no sky encoding
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_WRGBD, reason=f"WildRGB-D data not found at {WILDRGBD_DIR}")
def test_wildrgbd_getitem_tuple_index():
    ds = WildRgbdDataset(
        common_conf=_common_conf(),
        split="train",
        WILDRGBD_DIR=WILDRGBD_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_WRGBD, reason=f"WildRGB-D data not found at {WILDRGBD_DIR}")
def test_wildrgbd_splits_are_disjoint_and_sum():
    """train/test split indices are disjoint by scene and 'all' is their union
    (read straight from the on-disk index; no per-scene access)."""
    train = WildRgbdDataset.load_split_index(WILDRGBD_DIR, "train")
    test = WildRgbdDataset.load_split_index(WILDRGBD_DIR, "test")
    both = WildRgbdDataset.load_split_index(WILDRGBD_DIR, "all")
    assert set(train).isdisjoint(test)
    assert set(both) == set(train) | set(test)
    assert SEQ in train and SEQ2 in train


@pytest.mark.skipif(not HAVE_WRGBD, reason=f"WildRGB-D data not found at {WILDRGBD_DIR}")
def test_wildrgbd_reprojection_closure():
    """World points from two frames must be mutually consistent: frame A's valid
    world points reprojected into frame B land at depths matching B's depth map
    (locks depth-scale x pose-convention x intrinsics consistency end-to-end
    through process_one_image).

    The pair must be WIDE-baseline to discriminate conventions: at tiny baselines
    every pose hypothesis (no-invert, OpenGL flip, wrong depth scale) also closes
    under the thresholds below. ids [0, 24] of apple/scene_002 is a 0.304 m
    baseline: the correct c2w-OpenCV convention closes at ~0.6% median while the
    wrong hypotheses give >7% / 0 valid pixels / ~30%."""
    ds = WildRgbdDataset(
        common_conf=_eval_common(),
        split="train",
        WILDRGBD_DIR=WILDRGBD_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    b = ds.get_data(seq_name=ds.sequence_list[0], ids=np.array([0, 24]), aspect_ratio=1.0)
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


@pytest.mark.skipif(not HAVE_WRGBD, reason=f"WildRGB-D data not found at {WILDRGBD_DIR}")
def test_wildrgbd_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _wildrgbd_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" in sample                   # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_WRGBD, reason=f"WildRGB-D data not found at {WILDRGBD_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _wildrgbd_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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
    seq_name = composed.sequence_name(0)
    batch = vendor.get_data(seq_name=seq_name, ids=np.array(ids), aspect_ratio=0.75)
    manual = (
        torch.from_numpy(np.stack(batch["images"]).astype(np.float32))
        .permute(0, 3, 1, 2)
        .to(torch.get_default_dtype())
        .div(255)
    )
    torch.testing.assert_close(sample["images"], manual)

    # Order honored verbatim: each returned extrinsic equals the w2c read straight
    # from that frame's metadata npz, in the requested (unordered) id order. (In
    # eval mode process_one_image never alters extrinsics: no aug, no rotation.)
    expected = []
    for fi in ids:
        meta_path = vendor.frame_paths(seq_name, vendor.data_store[seq_name][fi])[2]
        expected.append(WildRgbdDataset.read_wildrgbd_metadata(meta_path)[1])
    expected = np.stack(expected)
    # the requested frames really have distinct poses (order check is not vacuous)
    assert np.ptp(expected[:, :, 3], axis=0).max() > 1e-3
    np.testing.assert_allclose(sample["extrinsics"].numpy(), expected, atol=1e-6)


@pytest.mark.skipif(not HAVE_WRGBD, reason=f"WildRGB-D data not found at {WILDRGBD_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ, SEQ2]
    composed = instantiate(
        _wildrgbd_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name])
        assert composed.sequence_num_frames(gi) == 100  # every WildRGB-D scene has 100 frames


@pytest.mark.skipif(not HAVE_WRGBD, reason=f"WildRGB-D data not found at {WILDRGBD_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants. WildRGB-D is portrait
    (H 512, W 386 for this scene; W is not /16) so the target snaps to patch-
    friendly shapes via get_target_shape."""
    from hydra.utils import instantiate

    composed = instantiate(
        _wildrgbd_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (512, 386)                       # native portrait (H, W)

    composed.set_img_size(384)                        # near-native short side, /16
    assert composed.img_size == 384
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (496, 384)   # int(384*512/386)=509 -> /16 -> 496

    composed.set_img_size(256)                        # smaller short side
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (336, 256)  # int(256*512/386)=339 -> /16 -> 336
