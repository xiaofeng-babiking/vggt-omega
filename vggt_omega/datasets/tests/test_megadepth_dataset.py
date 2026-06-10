import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.megadepth import MegaDepthDataset

MEGADEPTH_DIR = "/jfs/Data_4DFF/train_data/megadepth"
HAVE_MD = os.path.isdir(MEGADEPTH_DIR)
# Pinned sequences (verified on disk):
#  - 0000/0: 2186-frame Flickr scene, focal length varies per frame.
#  - 5017/0: 521-frame sequential DSLR scene, adjacent frames strongly covisible.
SEQ_FLICKR = "0000/0"
SEQ_DSLR = "5017/0"
# A verified degenerate "ordinal" depth frame (every valid pixel == 2.0).
DEGENERATE_EXR = os.path.join(MEGADEPTH_DIR, "0323/0/1002262331_5ee05e13ff_o.jpg.exr")


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


def _md_dataset_cfg(seqs=(SEQ_FLICKR,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.megadepth.MegaDepthDataset",
                "split": "train",
                "MEGADEPTH_DIR": MEGADEPTH_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- MegaDepth-specific helper unit tests (no data required) ---


def test_c2w_to_w2c_identity_and_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = MegaDepthDataset.megadepth_c2w_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_c2w_to_w2c_rotation_inverse():
    # camera-to-world: 90-degree rotation about z, translated camera center
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = MegaDepthDataset.megadepth_c2w_to_w2c(c2w)
    np.testing.assert_allclose(w2c[:3, :3], rot.T, atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], -rot.T @ [1.0, 2.0, 3.0], atol=1e-6)
    # the camera center must map to the camera-frame origin
    np.testing.assert_allclose(w2c[:3, :3] @ c2w[:3, 3] + w2c[:, 3], 0.0, atol=1e-6)


def test_c2w_to_w2c_rejects_bad_input():
    with pytest.raises(ValueError, match="non-finite"):
        MegaDepthDataset.megadepth_c2w_to_w2c(np.full((4, 4), np.inf))
    with pytest.raises(ValueError, match="expected"):
        MegaDepthDataset.megadepth_c2w_to_w2c(np.eye(3))


def test_read_megadepth_camera_synthetic(tmp_path):
    from safetensors.numpy import save_file

    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]])
    p = tmp_path / "frame.jpg.safetensor"
    save_file({"cam2world": c2w, "intrinsics": K}, str(p))
    w2c, k = MegaDepthDataset.read_megadepth_camera(str(p))
    assert w2c.dtype == np.float32 and k.dtype == np.float32
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)
    np.testing.assert_allclose(k, K, atol=1e-6)

    bad = tmp_path / "bad.jpg.safetensor"
    save_file({"cam2world": c2w, "intrinsics": np.zeros((3, 3))}, str(bad))
    with pytest.raises(ValueError, match="intrinsics"):
        MegaDepthDataset.read_megadepth_camera(str(bad))

    nonfinite = tmp_path / "nan.jpg.safetensor"
    save_file({"cam2world": np.full((4, 4), np.nan), "intrinsics": K}, str(nonfinite))
    with pytest.raises(ValueError, match="non-finite"):
        MegaDepthDataset.read_megadepth_camera(str(nonfinite))


def test_depth_decode_values_and_invalid(tmp_path):
    import cv2

    arr = np.array([[0.0, 1.5], [2.5, -3.0]], dtype=np.float32)
    p = tmp_path / "frame.jpg.exr"
    assert cv2.imwrite(str(p), arr)
    # min_depth_unique=0 disables the ordinal filter: raw value mapping only
    depth = MegaDepthDataset.read_megadepth_depth(str(p), min_depth_unique=0)
    assert depth.dtype == np.float32
    np.testing.assert_allclose(depth, [[0.0, 1.5], [2.5, 0.0]])  # 0 stays 0, negative -> 0


def test_depth_decode_ordinal_filter(tmp_path):
    import cv2

    # degenerate "ordinal" frame: every valid pixel == 2.0 -> zeroed by default
    deg = np.full((4, 4), 2.0, dtype=np.float32)
    deg[0, 0] = 0.0
    p = tmp_path / "deg.jpg.exr"
    assert cv2.imwrite(str(p), deg)
    assert (MegaDepthDataset.read_megadepth_depth(str(p)) == 0).all()
    assert (MegaDepthDataset.read_megadepth_depth(str(p), min_depth_unique=0) > 0).sum() == 15

    # rich depth (>= 5 unique positive values) passes through untouched
    rich = np.arange(16, dtype=np.float32).reshape(4, 4)
    p2 = tmp_path / "rich.jpg.exr"
    assert cv2.imwrite(str(p2), rich)
    np.testing.assert_allclose(MegaDepthDataset.read_megadepth_depth(str(p2)), rich)

    with pytest.raises(FileNotFoundError):
        MegaDepthDataset.read_megadepth_depth(str(tmp_path / "missing.jpg.exr"))


def test_sequence_matches_semantics():
    m = MegaDepthDataset.sequence_matches
    assert m("0000", "0", ["*"])
    assert m("0000", "0", ["0000"])          # scene-only pattern: all subs
    assert m("0000", "1", ["0000"])
    assert m("0000", "0", ["00*"])
    assert m("0000", "0", ["0000/0"])        # exact sequence
    assert not m("0000", "1", ["0000/0"])
    assert not m("0001", "0", ["0000"])
    assert m("0323", "1", ["*/1"])           # sub-only glob
    assert not m("0323", "0", ["*/1"])
    assert m("0001", "0", ["0000/1", "00*"])  # any pattern may match


# --- MegaDepth integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_megadepth_sample_schema_and_conventions():
    ds = MegaDepthDataset(
        common_conf=_common_conf(),
        split="train",
        MEGADEPTH_DIR=MEGADEPTH_DIR,
        sequences=[SEQ_FLICKR],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.SKY_MASK not in ds.available_modalities   # no sky labels (sky folded into 0)
    assert Modality.TIMESTAMP not in ds.available_modalities  # unordered photos, no clock

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
    assert (depth >= 0).all()                             # 0=invalid incl. sky; never negative
    assert (depth[depth > 0]).size > 0                    # some valid (SfM-scale) depth
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["seq_name"] == "megadepth_" + SEQ_FLICKR
    assert batch["is_metric"] is False and batch["is_video"] is False
    assert "timestamps" not in batch                      # not advertised, not fabricated
    # Same rule for sky: outdoor photos DO contain sky (folded into depth==0),
    # so an all-False sky_masks would be wrong GT -- the key must be absent
    # (carry_extra_modalities tensorizes any registered key present in batch).
    assert "sky_masks" not in batch

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_megadepth_reprojection_closure():
    """Depth-scale x pose-convention x intrinsics lock: frame A's valid world
    points reprojected into covisible frame B must land at depths matching B's
    depth map (relative error, since MegaDepth depth is SfM-scale not meters).
    Frames 0 and 1 of the sequential DSLR scene 5017/0 are strongly covisible."""
    ds = MegaDepthDataset(
        common_conf=_eval_common(),
        split="train",
        MEGADEPTH_DIR=MEGADEPTH_DIR,
        sequences=[SEQ_DSLR],
        len_train=10,
    )
    b = ds.get_data(seq_name=SEQ_DSLR, ids=np.array([0, 1]), aspect_ratio=0.75)
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
    # Threshold chosen to discriminate conventions on this exact pair: the
    # correct OpenCV c2w rigid inverse gives ~0.001, a WRONG OpenGL axis flip
    # (c2w @ diag(1,-1,-1,1) before inverting) gives ~0.049, and no inversion
    # at all gives ~0.17. 0.01 keeps ~10x headroom over correct while failing
    # both wrong conventions (0.05 would let the OpenGL flip squeak through).
    assert np.median(rel_err) < 0.01


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_megadepth_degenerate_frame_on_disk_is_zeroed():
    """A real pinned ordinal-depth frame (scene 0323) is zeroed by the default
    filter but decodes to the constant 2.0 with the filter disabled."""
    assert (MegaDepthDataset.read_megadepth_depth(DEGENERATE_EXR) == 0).all()
    raw = MegaDepthDataset.read_megadepth_depth(DEGENERATE_EXR, min_depth_unique=0)
    pos = raw[raw > 0]
    assert pos.size > 0 and np.unique(pos).size == 1 and pos[0] == 2.0


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_megadepth_getitem_tuple_index():
    ds = MegaDepthDataset(
        common_conf=_common_conf(),
        split="train",
        MEGADEPTH_DIR=MEGADEPTH_DIR,
        sequences=[SEQ_FLICKR],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_megadepth_lazy_full_enumeration():
    """Unfiltered construction enumerates all 210 scene-sub sequences from DISK
    (the root npz index lists 43 sequences that do not exist) without touching
    any per-frame files: frame lists are built lazily, one sequence at a time."""
    import time

    t0 = time.time()
    ds = MegaDepthDataset(
        common_conf=_common_conf(),
        split="train",
        MEGADEPTH_DIR=MEGADEPTH_DIR,
        len_train=10,
    )
    elapsed = time.time() - t0
    assert ds.sequence_list_len == 210
    assert elapsed < 60, f"construction took {elapsed:.1f}s; expected lazy (a few seconds)"
    assert ds._frames_cache == {}                  # nothing enumerated eagerly

    li = ds.sequence_list.index(SEQ_DSLR)
    assert ds.sequence_num_frames(li) == 521       # verified frame count on disk
    assert set(ds._frames_cache) == {SEQ_DSLR}     # only the accessed sequence was listed
    assert ds.native_image_size(li) == (600, 803)  # (H, W) of the first frame


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_megadepth_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _md_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "modalities" in sample
    assert "timestamps" not in sample              # TIMESTAMP not advertised: nothing fabricated
    assert "sky_masks" not in sample               # SKY_MASK not advertised: nothing fabricated
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _md_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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

    # Order honored: MegaDepth has no timestamps, but the Flickr scene's focal
    # length differs per frame -- a per-frame quantity that must follow the
    # requested (unordered) id order, both vs get_data and under id reversal.
    fx = sample["intrinsics"][:, 0, 0]
    assert len(torch.unique(fx)) > 1                        # non-vacuous: focals really differ
    torch.testing.assert_close(
        sample["intrinsics"], torch.from_numpy(np.stack(batch["intrinsics"]))
    )
    rev = composed.get_sample(0, ids=ids[::-1], aspect_ratio=0.75)
    torch.testing.assert_close(rev["images"], sample["images"].flip(0))
    torch.testing.assert_close(rev["intrinsics"], sample["intrinsics"].flip(0))


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ_FLICKR, SEQ_DSLR]
    composed = instantiate(
        _md_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor._frames(name))


@pytest.mark.skipif(not HAVE_MD, reason=f"MegaDepth data not found at {MEGADEPTH_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _md_dataset_cfg(seqs=[SEQ_DSLR]), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (600, 803)                       # 5017/0 first frame (H, W); short side 600

    composed.set_img_size(640)
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    # short side int(640*600/803)=478 snapped down to /16 -> 464
    assert tuple(s["images"].shape[-2:]) == (464, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (224, 320)
