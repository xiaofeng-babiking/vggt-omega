import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.sintel import SintelDataset

SINTEL_DIR = "/jfs/guibiao/streamVGGT/data/eval/sintel"
HAVE_SINTEL = os.path.isdir(os.path.join(SINTEL_DIR, "training"))
# alley_1 is a 50-frame, largely static sequence; use it for fast integration tests.
SEQ = "alley_1"


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


def _sintel_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.sintel.SintelDataset",
                "split": "train",
                "SINTEL_DIR": SINTEL_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- synthetic PIEH-binary writers for data-free unit tests ---

_TAG = np.float32(202021.25)


def _write_dpt(path, depth):
    """Write a Sintel .dpt: float32 PIEH tag, int32 width, int32 height, float32 data."""
    depth = np.asarray(depth, dtype=np.float32)
    h, w = depth.shape
    with open(path, "wb") as f:
        _TAG.tofile(f)
        np.array([w, h], dtype=np.int32).tofile(f)
        depth.tofile(f)


def _write_cam(path, K, w2c):
    """Write a Sintel .cam: float32 PIEH tag, 9 float64 K, 12 float64 extrinsic."""
    with open(path, "wb") as f:
        _TAG.tofile(f)
        np.asarray(K, dtype=np.float64).reshape(9).tofile(f)
        np.asarray(w2c, dtype=np.float64).reshape(12).tofile(f)


# --- Sintel-specific helper unit tests (no data required) ---


def test_depth_reader_decodes_meters_and_maps_sky(tmp_path):
    p = tmp_path / "frame_0001.dpt"
    _write_dpt(p, [[1.5, 2.0], [999.0, 1e11]])
    depth = SintelDataset.read_sintel_depth(str(p))
    assert depth.dtype == np.float32 and depth.shape == (2, 2)
    # values below the default 1000 m threshold pass through; sky -> -1.0
    np.testing.assert_allclose(depth, [[1.5, 2.0], [999.0, -1.0]])
    # custom threshold pulls the 999 m sky-dome pixel in too
    depth2 = SintelDataset.read_sintel_depth(str(p), sky_threshold=500.0)
    np.testing.assert_allclose(depth2, [[1.5, 2.0], [-1.0, -1.0]])


def test_depth_reader_maps_non_finite_and_negative_to_invalid(tmp_path):
    p = tmp_path / "frame_0001.dpt"
    _write_dpt(p, [[np.inf, np.nan], [-3.0, 4.0]])
    depth = SintelDataset.read_sintel_depth(str(p))
    np.testing.assert_allclose(depth, [[0.0, 0.0], [0.0, 4.0]])


def test_depth_reader_rejects_bad_tag(tmp_path):
    p = tmp_path / "bad.dpt"
    with open(p, "wb") as f:
        np.float32(123.0).tofile(f)
        np.array([2, 2], dtype=np.int32).tofile(f)
        np.zeros(4, dtype=np.float32).tofile(f)
    with pytest.raises(ValueError, match="tag"):
        SintelDataset.read_sintel_depth(str(p))


def test_depth_reader_rejects_truncated_payload(tmp_path):
    p = tmp_path / "trunc.dpt"
    with open(p, "wb") as f:
        _TAG.tofile(f)
        np.array([4, 4], dtype=np.int32).tofile(f)
        np.zeros(7, dtype=np.float32).tofile(f)  # 7 of 16 values
    with pytest.raises(ValueError, match="truncated"):
        SintelDataset.read_sintel_depth(str(p))


def test_cam_reader_returns_w2c_directly_no_inversion(tmp_path):
    # 90-degree rotation about z + translation: if the reader (wrongly) inverted
    # the stored matrix, the translation would come back as -R^T t = (2, -1, -3).
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    w2c_in = np.hstack([R, t[:, None]])
    K_in = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 25.0], [0.0, 0.0, 1.0]])
    p = tmp_path / "frame_0001.cam"
    _write_cam(p, K_in, w2c_in)
    K, w2c = SintelDataset.read_sintel_cam(str(p))
    assert K.dtype == np.float32 and K.shape == (3, 3)
    assert w2c.dtype == np.float32 and w2c.shape == (3, 4)
    np.testing.assert_allclose(K, K_in, atol=1e-6)
    np.testing.assert_allclose(w2c, w2c_in, atol=1e-6)  # used as-is: stored = w2c


def test_cam_reader_rejects_bad_tag_truncation_and_non_finite(tmp_path):
    bad_tag = tmp_path / "bad.cam"
    with open(bad_tag, "wb") as f:
        np.float32(0.0).tofile(f)
        np.zeros(21, dtype=np.float64).tofile(f)
    with pytest.raises(ValueError, match="tag"):
        SintelDataset.read_sintel_cam(str(bad_tag))

    trunc = tmp_path / "trunc.cam"
    with open(trunc, "wb") as f:
        _TAG.tofile(f)
        np.zeros(15, dtype=np.float64).tofile(f)  # 15 of 21 values
    with pytest.raises(ValueError, match="truncated"):
        SintelDataset.read_sintel_cam(str(trunc))

    nonfinite = tmp_path / "inf.cam"
    K = np.eye(3) * 100.0
    K[2, 2] = 1.0
    w2c = np.hstack([np.eye(3), np.full((3, 1), np.inf)])
    _write_cam(nonfinite, K, w2c)
    with pytest.raises(ValueError, match="non-finite"):
        SintelDataset.read_sintel_cam(str(nonfinite))


def test_constructor_validates_render_pass(tmp_path):
    with pytest.raises(ValueError, match="render_pass"):
        SintelDataset(
            common_conf=_common_conf(), SINTEL_DIR=str(tmp_path), render_pass="albedo"
        )


# --- Sintel integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_sample_schema_and_conventions():
    ds = SintelDataset(
        common_conf=_common_conf(),
        split="train",
        SINTEL_DIR=SINTEL_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.SKY_MASK in ds.available_modalities
    assert Modality.TIMESTAMP not in ds.available_modalities

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
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert sky.dtype == bool and sky.shape == depth.shape
    assert not (sky & pmask).any()                        # sky never counts as valid depth
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_sky_sentinel_yields_true_sky_mask():
    """market_2 encodes sky as the ~1e11 sentinel (~0.5% of pixels): the loader
    must mark those pixels sky (depth < 0) and exclude them from point_masks."""
    ds = SintelDataset(
        common_conf=_eval_common(),
        split="train",
        SINTEL_DIR=SINTEL_DIR,
        sequences=["market_2"],
        len_train=10,
    )
    b = ds.get_data(seq_name="market_2", ids=np.array([0, 1]), aspect_ratio=0.75)
    depth = np.stack(b["depths"])
    sky = np.stack(b["sky_masks"])
    pmask = np.stack(b["point_masks"])
    assert sky.any()                                      # real sky pixels survive resize
    assert (depth[sky] < 0).all()
    assert not (sky & pmask).any()
    assert (depth[~sky] >= 0).all()


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_reprojection_closure_proves_w2c_convention():
    """World points from two well-separated frames of a static scene must be
    mutually consistent: frame A's valid world points reprojected into frame B
    land at depths matching B's depth map. temple_2 f1->f30 closes at ~0.06%
    median relative error with the stored matrix used directly as w2c, but at
    ~41% if misread as camera-to-world -- so the 1% bound below decisively
    FAILS under a flipped convention."""
    ds = SintelDataset(
        common_conf=_eval_common(),
        split="train",
        SINTEL_DIR=SINTEL_DIR,
        sequences=["temple_2"],
        len_train=10,
    )
    b = ds.get_data(seq_name="temple_2", ids=np.array([0, 29]), aspect_ratio=0.75)
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
    assert valid.sum() > 500
    assert np.median(rel_err) < 0.01  # tight: a flipped convention sits at ~0.41


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_render_passes_share_geometry():
    """clean and final share depth/ and camdata_left/: identical depth, K and
    poses for the same frames, but different RGB (final adds blur/fog/effects)."""
    kwargs = dict(
        common_conf=_eval_common(),
        split="train",
        SINTEL_DIR=SINTEL_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    ds_final = SintelDataset(render_pass="final", **kwargs)   # the default protocol
    ds_clean = SintelDataset(render_pass="clean", **kwargs)
    assert ds_final.sequence_list == ds_clean.sequence_list
    assert ds_final.sequence_num_frames(0) == ds_clean.sequence_num_frames(0)

    ids = np.array([0, 5])
    bf = ds_final.get_data(seq_name=SEQ, ids=ids, aspect_ratio=0.75)
    bc = ds_clean.get_data(seq_name=SEQ, ids=ids, aspect_ratio=0.75)
    np.testing.assert_allclose(np.stack(bf["depths"]), np.stack(bc["depths"]))
    np.testing.assert_allclose(np.stack(bf["extrinsics"]), np.stack(bc["extrinsics"]))
    np.testing.assert_allclose(np.stack(bf["intrinsics"]), np.stack(bc["intrinsics"]))
    assert not np.array_equal(np.stack(bf["images"]), np.stack(bc["images"]))


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_short_real_sequences_are_kept():
    """min_num_images defaults to 8 so the real short scenes (ambush_6 = 20
    frames, ambush_2 = 21) are NOT dropped (eval dataset: dropping real
    sequences is worse than short sequences)."""
    ds = SintelDataset(
        common_conf=_common_conf(),
        split="train",
        SINTEL_DIR=SINTEL_DIR,
        sequences=["ambush_2", "ambush_6"],
        len_train=10,
    )
    assert set(ds.sequence_list) == {"ambush_2", "ambush_6"}
    counts = {name: len(ds.data_store[name]) for name in ds.sequence_list}
    assert counts == {"ambush_2": 21, "ambush_6": 20}


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_getitem_tuple_index():
    ds = SintelDataset(
        common_conf=_common_conf(),
        split="train",
        SINTEL_DIR=SINTEL_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _sintel_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" in sample                   # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _sintel_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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
    # Order honored: per-frame poses follow the requested (unordered) id order.
    np.testing.assert_allclose(
        sample["extrinsics"].numpy(), np.stack(batch["extrinsics"]), atol=1e-6
    )
    single = vendor.get_data(
        seq_name=composed.sequence_name(0), ids=np.array([ids[1]]), aspect_ratio=0.75
    )
    np.testing.assert_allclose(
        sample["extrinsics"][1].numpy(), single["extrinsics"][0], atol=1e-6
    )


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ, "ambush_6"]
    composed = instantiate(
        _sintel_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name])


@pytest.mark.skipif(not HAVE_SINTEL, reason=f"Sintel data not found at {SINTEL_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _sintel_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (436, 1024)                      # Sintel native (H, W)

    composed.set_img_size(1024)                       # native long side
    assert composed.img_size == 1024
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    # 436 is not /16-divisible: get_target_shape snaps the short side to 432
    assert tuple(s["images"].shape[-2:]) == (432, 1024)

    composed.set_img_size(512)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (208, 512)
