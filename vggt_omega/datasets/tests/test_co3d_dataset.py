import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.co3d import Co3dDataset

CO3D_DIR = "/jfs/Data_4DFF/train_data/co3d"
HAVE_CO3D = os.path.isdir(CO3D_DIR)
# Survey-verified sequence (pose convention gold-checked by reprojection).
SEQ = "apple/110_13072_25709"
SEQ2 = "apple/151_16773_32218"


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


def _co3d_dataset_cfg(seqs=(SEQ,), n=20, split="train"):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.co3d.Co3dDataset",
                "split": split,
                "CO3D_DIR": CO3D_DIR,
                "sequences": list(seqs),
                "len_train": n,
                "len_test": n,
            }
        ],
    }


# --- CO3D-specific helper unit tests (no data required) ---


def test_co3d_pose_to_w2c_identity_and_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = Co3dDataset.co3d_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_co3d_pose_to_w2c_rotation_roundtrip():
    # 90-degree rotation about z plus translation: w2c must exactly invert c2w.
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    c2w = np.eye(4)
    c2w[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    c2w[:3, 3] = [0.5, -1.5, 2.0]
    w2c = Co3dDataset.co3d_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3] @ c2w[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:3, :3] @ c2w[:3, 3] + w2c[:, 3], 0.0, atol=1e-6)


def test_co3d_pose_to_w2c_rejects_bad():
    with pytest.raises(ValueError, match="shape"):
        Co3dDataset.co3d_pose_to_w2c(np.eye(3))
    bad = np.eye(4)
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        Co3dDataset.co3d_pose_to_w2c(bad)


def test_co3d_depth_decode_units_and_invalid(tmp_path):
    from PIL import Image

    # 13107/65535 == 0.2 exactly; 0 is the (only) invalid encoding and stays 0.
    arr = np.array([[0, 65535], [13107, 32768]], dtype=np.uint16)
    p = tmp_path / "frame000001.jpg.geometric.png"
    Image.fromarray(arr).save(p)
    depth = Co3dDataset.read_co3d_depth(str(p), max_depth=2.0)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(
        depth, [[0.0, 2.0], [0.4, 2.0 * 32768 / 65535]], rtol=1e-6
    )
    # Same PNG, different per-frame scale -> different depths (per-frame normalized).
    depth10 = Co3dDataset.read_co3d_depth(str(p), max_depth=10.0)
    np.testing.assert_allclose(depth10, depth * 5.0, rtol=1e-6)


def test_co3d_frame_meta_roundtrip_and_errors(tmp_path):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    K = np.array([[700.0, 0.0, 192.0], [0.0, 700.0, 341.0], [0.0, 0.0, 1.0]])
    good = tmp_path / "frame000001.npz"
    np.savez(good, camera_pose=c2w, camera_intrinsics=K, maximum_depth=np.float32(46.4))
    c2w_r, K_r, md = Co3dDataset.read_co3d_frame_meta(str(good))
    np.testing.assert_allclose(c2w_r, c2w, atol=1e-6)
    np.testing.assert_allclose(K_r, K, atol=1e-6)
    assert md == pytest.approx(46.4, rel=1e-5)

    missing = tmp_path / "missing_pose.npz"
    np.savez(missing, camera_intrinsics=K, maximum_depth=1.0)
    with pytest.raises(ValueError, match="missing key"):
        Co3dDataset.read_co3d_frame_meta(str(missing))

    bad_scale = tmp_path / "bad_scale.npz"
    np.savez(bad_scale, camera_pose=c2w, camera_intrinsics=K, maximum_depth=0.0)
    with pytest.raises(ValueError, match="maximum_depth"):
        Co3dDataset.read_co3d_frame_meta(str(bad_scale))


def test_co3d_intrinsics_native_override_and_error():
    K_native = np.array([[712.6, 0.0, 191.7], [0.0, 712.6, 341.2], [0.0, 0.0, 1.0]])
    K = Co3dDataset.co3d_intrinsics(K_native)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, K_native, rtol=1e-6)
    K2 = Co3dDataset.co3d_intrinsics(K_native, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0
    with pytest.raises(ValueError, match="shape"):
        Co3dDataset.co3d_intrinsics(np.eye(4))


def test_co3d_frame_paths_embed_rgb_name():
    rgb, depth, npz = Co3dDataset.co3d_frame_paths("/root/apple/110", 7)
    assert rgb == "/root/apple/110/images/frame000007.jpg"
    # the depth filename embeds the rgb name (frame{i}.jpg.geometric.png)
    assert depth == "/root/apple/110/depths/frame000007.jpg.geometric.png"
    assert npz == "/root/apple/110/images/frame000007.npz"


# --- CO3D integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_co3d_sample_schema_and_conventions():
    ds = Co3dDataset(
        common_conf=_common_conf(),
        split="train",
        CO3D_DIR=CO3D_DIR,
        sequences=[SEQ],
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
    assert (depth[depth > 0]).size > 0                    # some valid depth
    assert (depth >= 0).all()                             # no sky encoding in CO3D
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is False                    # SfM-arbitrary scale
    assert batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_co3d_getitem_tuple_index():
    ds = Co3dDataset(
        common_conf=_common_conf(),
        split="train",
        CO3D_DIR=CO3D_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_co3d_split_is_frame_level():
    """CO3D's train/test jsons list the SAME sequences; the split is over frame
    indices (train list is a strict subset of the full/test list per sequence)."""
    train_ds = Co3dDataset(
        common_conf=_common_conf(), split="train",
        CO3D_DIR=CO3D_DIR, sequences=[SEQ], len_train=10,
    )
    test_ds = Co3dDataset(
        common_conf=_common_conf(), split="test",
        CO3D_DIR=CO3D_DIR, sequences=[SEQ], len_test=10,
    )
    assert train_ds.sequence_list == test_ds.sequence_list == [SEQ]
    train_frames = set(train_ds.data_store[SEQ])
    test_frames = set(test_ds.data_store[SEQ])
    assert train_frames < test_frames                     # strict frame-level subset
    assert train_ds.sequence_num_frames(0) < test_ds.sequence_num_frames(0)


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_co3d_sequences_filter_by_category_glob():
    """`sequences` patterns match the full "category/seq_id" name, the category
    alone, or the seq_id alone."""
    ds = Co3dDataset(
        common_conf=_common_conf(), split="train",
        CO3D_DIR=CO3D_DIR, sequences=["apple"], len_train=10,
    )
    assert len(ds.sequence_list) >= 2
    assert all(name.startswith("apple/") for name in ds.sequence_list)
    assert SEQ in ds.sequence_list and SEQ2 in ds.sequence_list
    with pytest.raises(ValueError, match="No usable CO3D sequences"):
        Co3dDataset(
            common_conf=_common_conf(), split="train",
            CO3D_DIR=CO3D_DIR, sequences=["no_such_category"], len_train=10,
        )


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_co3d_reprojection_closure():
    """World points from two frames of the same sequence must be mutually
    consistent: frame A's valid world points reprojected into frame B land at
    depths matching B's depth map (validates depth-scale x pose-convention x
    per-frame intrinsics agreement end-to-end through process_one_image).
    CO3D is SfM-scaled, so the error is RELATIVE."""
    ds = Co3dDataset(
        common_conf=_eval_common(),
        split="test",
        CO3D_DIR=CO3D_DIR,
        sequences=[SEQ],
        len_test=10,
    )
    b = ds.get_data(seq_name=ds.sequence_list[0], ids=np.array([0, 15]), aspect_ratio=1.0)
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


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_co3d_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _co3d_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" not in sample               # unadvertised -> key not emitted
    assert "timestamps" not in sample              # CO3D ships no timestamps
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _co3d_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]
    sample = composed.get_sample(0, ids=ids, aspect_ratio=1.0)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample

    # Drift guard: the same vendor.get_data + manual tensorize must match byte-for-byte.
    vendor = composed.base_dataset.datasets[0]
    batch = vendor.get_data(
        seq_name=composed.sequence_name(0), ids=np.array(ids), aspect_ratio=1.0
    )
    manual = (
        torch.from_numpy(np.stack(batch["images"]).astype(np.float32))
        .permute(0, 3, 1, 2)
        .to(torch.get_default_dtype())
        .div(255)
    )
    torch.testing.assert_close(sample["images"], manual)

    # Order honored verbatim: CO3D has no per-frame timestamps, so prove order
    # via per-frame extrinsics (every frame has a distinct pose): fetching each
    # id individually must reproduce the unordered batch row-for-row.
    for k, i in enumerate(ids):
        single = vendor.get_data(
            seq_name=composed.sequence_name(0), ids=np.array([i]), aspect_ratio=1.0
        )
        np.testing.assert_array_equal(
            sample["extrinsics"][k].numpy(), single["extrinsics"][0]
        )


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ, SEQ2]
    composed = instantiate(
        _co3d_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name])


@pytest.mark.skipif(not HAVE_CO3D, reason=f"CO3D data not found at {CO3D_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _co3d_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (683, 384)                       # this CO3D sequence is portrait

    # CO3D's get_target_shape uses img_size as the WIDTH and snaps the height to
    # the patch grid: img_size=384 (native width), aspect=h/w -> (672, 384).
    composed.set_img_size(384)
    assert composed.img_size == 384
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (672, 384)

    composed.set_img_size(192)                        # half-res, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (336, 192)
