import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.mvs_synth import MvsSynthDataset

MVS_SYNTH_DIR = "/jfs/Data_4DFF/train_data/mvs_synth"
HAVE_MVS = os.path.isdir(MVS_SYNTH_DIR)
# Single sequence keeps the integration tests fast (each seq has 100 frames).
SEQ = "0000"


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


def _mvs_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.mvs_synth.MvsSynthDataset",
                "split": "train",
                "MVS_SYNTH_DIR": MVS_SYNTH_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- MVS-Synth-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    w2c = MvsSynthDataset.mvs_synth_pose_to_w2c(np.eye(4))
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-6)

    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = MvsSynthDataset.mvs_synth_pose_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_axis_remap_round_trip():
    # c2w rotation = 90 deg about world z, translation (10, -5, 2): a camera-frame
    # point must round-trip world -> camera through the returned w2c.
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [10.0, -5.0, 2.0]
    w2c = MvsSynthDataset.mvs_synth_pose_to_w2c(c2w)

    p_cam = np.array([1.0, 2.0, 3.0])
    p_world = rot @ p_cam + c2w[:3, 3]
    np.testing.assert_allclose(w2c[:, :3] @ p_world + w2c[:, 3], p_cam, atol=1e-5)


def test_pose_to_w2c_rejects_bad_input():
    with pytest.raises(ValueError, match="non-finite"):
        MvsSynthDataset.mvs_synth_pose_to_w2c(np.full((4, 4), np.inf))
    with pytest.raises(ValueError, match=r"\(4, 4\)"):
        MvsSynthDataset.mvs_synth_pose_to_w2c(np.eye(3))


def test_depth_decode_maps_sky_to_negative(tmp_path):
    # 0 (converted EXR-inf sky), negatives and non-finite all become -1.0; valid
    # metric values pass through; float64 input downcasts to float32.
    arr = np.array([[0.0, 2.5], [-3.0, np.inf], [np.nan, 7800.0]], dtype=np.float64)
    p = tmp_path / "0000.npy"
    np.save(p, arr)
    depth = MvsSynthDataset.read_mvs_synth_depth(str(p))
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[-1.0, 2.5], [-1.0, -1.0], [-1.0, 7800.0]])

    np.save(tmp_path / "bad.npy", np.zeros(5, dtype=np.float32))  # 1-D: not a depth map
    with pytest.raises(ValueError, match="2-D"):
        MvsSynthDataset.read_mvs_synth_depth(str(tmp_path / "bad.npy"))


def test_camera_reader_and_validation(tmp_path):
    K = np.array([[579.0, 0.0, 480.0], [0.0, 579.0, 270.0], [0.0, 0.0, 1.0]], np.float32)
    c2w = np.eye(4)
    c2w[:3, 3] = [9509.0, -4709.0, -304.0]
    p = tmp_path / "0000.npz"
    np.savez(p, intrinsics=K, pose=c2w)
    K2, c2w2 = MvsSynthDataset.read_mvs_synth_camera(str(p))
    assert K2.dtype == np.float32 and c2w2.dtype == np.float64
    np.testing.assert_allclose(K2, K)
    np.testing.assert_allclose(c2w2, c2w)

    np.savez(tmp_path / "nokey.npz", intrinsics=K)  # missing "pose"
    with pytest.raises(ValueError, match="missing key"):
        MvsSynthDataset.read_mvs_synth_camera(str(tmp_path / "nokey.npz"))
    np.savez(tmp_path / "badk.npz", intrinsics=K[:2], pose=c2w)
    with pytest.raises(ValueError, match="intrinsics shape"):
        MvsSynthDataset.read_mvs_synth_camera(str(tmp_path / "badk.npz"))


def test_intrinsics_assembly_override_and_error():
    K_file = np.array([[578.9, 0.0, 480.0], [0.0, 578.9, 270.0], [0.0, 0.0, 1.0]])
    K = MvsSynthDataset.mvs_synth_intrinsics(K_file)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, K_file)

    K2 = MvsSynthDataset.mvs_synth_intrinsics(K_file, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0  # override wins over the file K

    with pytest.raises(ValueError, match="intrinsics"):
        MvsSynthDataset.mvs_synth_intrinsics(None)
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        MvsSynthDataset.mvs_synth_intrinsics(np.eye(2))


# --- MVS-Synth integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_MVS, reason=f"MVS-Synth data not found at {MVS_SYNTH_DIR}")
def test_construction_is_lazy_and_fast():
    """Construction over ALL 120 sequences only lists names (one scandir): no
    per-sequence frame table may be materialized before first access."""
    ds = MvsSynthDataset(
        common_conf=_common_conf(), split="train", MVS_SYNTH_DIR=MVS_SYNTH_DIR, len_train=10,
    )
    assert ds.sequence_list_len == 120
    assert ds._frames_cache == {}                 # frame enumeration deferred
    assert ds.sequence_num_frames(0) == 100       # lazily filled on demand
    assert set(ds._frames_cache) == {ds.sequence_list[0]}


@pytest.mark.skipif(not HAVE_MVS, reason=f"MVS-Synth data not found at {MVS_SYNTH_DIR}")
def test_sample_schema_and_conventions():
    ds = MvsSynthDataset(
        common_conf=_common_conf(),
        split="train",
        MVS_SYNTH_DIR=MVS_SYNTH_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.SKY_MASK in ds.available_modalities

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
    sky = np.stack(batch["sky_masks"])

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert sky.any()                                      # GTA-V outdoor: sky present
    np.testing.assert_array_equal(sky, depth < 0)         # sky sentinel survives resize
    assert not pmask[sky].any()                           # sky never counts as valid depth
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["seq_name"] == "mvs_synth_" + SEQ
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_MVS, reason=f"MVS-Synth data not found at {MVS_SYNTH_DIR}")
def test_cross_frame_reprojection_closure():
    """REPROJECTION CLOSURE: frame A's valid world points reprojected into a
    nearby frame B land at depths matching B's depth map (locks depth scale x
    pose convention x intrinsics consistency end-to-end through
    process_one_image). Relative error, since depth spans 1 m .. ~7800 m."""
    ds = MvsSynthDataset(
        common_conf=_eval_common(),
        split="train",
        MVS_SYNTH_DIR=MVS_SYNTH_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ, ids=np.array([0, 2]), aspect_ratio=540 / 960)
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


@pytest.mark.skipif(not HAVE_MVS, reason=f"MVS-Synth data not found at {MVS_SYNTH_DIR}")
def test_getitem_tuple_index():
    ds = MvsSynthDataset(
        common_conf=_common_conf(),
        split="train",
        MVS_SYNTH_DIR=MVS_SYNTH_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_MVS, reason=f"MVS-Synth data not found at {MVS_SYNTH_DIR}")
def test_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _mvs_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" in sample                   # extended modality carried through
    assert sample["sky_masks"].dtype.is_floating_point is False
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_MVS, reason=f"MVS-Synth data not found at {MVS_SYNTH_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _mvs_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]                                     # deliberately unordered
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

    # Order honored: MVS-Synth has no timestamps, so prove order with a per-frame
    # quantity instead -- each returned extrinsic must equal the w2c pose of the
    # requested frame id, recomputed straight from that frame's cam file.
    frames = vendor._list_frames(composed.sequence_name(0))
    for k, fid in enumerate(ids):
        _, c2w = MvsSynthDataset.read_mvs_synth_camera(frames[fid][2])
        w2c = MvsSynthDataset.mvs_synth_pose_to_w2c(c2w)
        np.testing.assert_allclose(
            sample["extrinsics"][k].numpy(), w2c, rtol=1e-6, atol=1e-3
        )
    # ... and frames 3 vs 7 really have distinct poses (the check is not vacuous).
    assert not np.allclose(
        sample["extrinsics"][1].numpy(), sample["extrinsics"][2].numpy(), atol=1e-3
    )


@pytest.mark.skipif(not HAVE_MVS, reason=f"MVS-Synth data not found at {MVS_SYNTH_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = ["0000", "0001"]
    composed = instantiate(
        _mvs_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == 100     # survey: exactly 100/seq
        assert composed.sequence_num_frames(gi) == len(vendor._list_frames(name))


@pytest.mark.skipif(not HAVE_MVS, reason=f"MVS-Synth data not found at {MVS_SYNTH_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _mvs_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (540, 960)                       # MVS-Synth native (H, W)

    composed.set_img_size(960)                        # native long side
    assert composed.img_size == 960
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    # 540 is not divisible by patch_size 16, so the short side snaps down to 528.
    assert tuple(s["images"].shape[-2:]) == (528, 960)

    composed.set_img_size(480)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (256, 480)
