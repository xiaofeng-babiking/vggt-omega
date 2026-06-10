import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.omniobject3d import OmniObject3dDataset

OMNIOBJECT3D_DIR = "/jfs/Data_4DFF/train_data/omniobject3d"
HAVE_OO3D = os.path.isdir(OMNIOBJECT3D_DIR)
# Single small objects keep integration construction + loading fast.
SEQ = "anise/anise_001"
SEQ2 = "anise/anise_002"
# Adjacent-on-sphere pair the pose convention was verified on (57 deg apart,
# median reprojection rel err ~0.0009 at native res).
REPROJ_SEQ = "toy_plane/toy_plane_001"
REPROJ_IDS = (50, 51)


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


def _oo3d_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.omniobject3d.OmniObject3dDataset",
                "split": "train",
                "OMNIOBJECT3D_DIR": OMNIOBJECT3D_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- OmniObject3D-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    # camera-to-world = pure translation by (1,2,3), identity rotation
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = OmniObject3dDataset.omniobject3d_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4)
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_axis_remap_roundtrip():
    # camera-to-world with a 90-degree axis remap (x->y, y->-x) + translation:
    # a camera-frame point must round-trip world->camera exactly.
    rot_c2w = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    c2w[:3, :3] = rot_c2w
    c2w[:3, 3] = [0.5, -1.5, 4.0]
    w2c = OmniObject3dDataset.omniobject3d_pose_to_w2c(c2w)
    p_cam = np.array([0.3, -0.7, 2.0])
    p_world = rot_c2w @ p_cam + c2w[:3, 3]
    back = w2c[:3, :3] @ p_world + w2c[:, 3]
    np.testing.assert_allclose(back, p_cam, atol=1e-5)


def test_pose_to_w2c_rejects_bad_inputs():
    with pytest.raises(ValueError, match="non-finite"):
        OmniObject3dDataset.omniobject3d_pose_to_w2c(np.full((4, 4), np.inf))
    with pytest.raises(ValueError, match=r"\(4,4\)"):
        OmniObject3dDataset.omniobject3d_pose_to_w2c(np.eye(3))


def test_depth_reader_invalid_and_negative(tmp_path):
    arr = np.array([[0.0, 1.5], [np.nan, -0.25]], dtype=np.float32)
    p = tmp_path / "r_0.npy"
    np.save(p, arr)
    depth = OmniObject3dDataset.read_omniobject3d_depth(str(p))
    assert depth.dtype == np.float32
    # 0 stays invalid; nan and negative (reserved for sky, absent here) -> 0
    np.testing.assert_allclose(depth, [[0.0, 1.5], [0.0, 0.0]])
    np.save(p, np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        OmniObject3dDataset.read_omniobject3d_depth(str(p))


def test_cam_reader_and_errors(tmp_path):
    K = np.array([[1111.1111, 0, 400.0], [0, 1111.1111, 400.0], [0, 0, 1.0]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    p = tmp_path / "r_0.npz"
    np.savez(p, intrinsics=K, pose=c2w)
    K_out, c2w_out = OmniObject3dDataset.read_omniobject3d_cam(str(p))
    np.testing.assert_allclose(K_out, K)
    np.testing.assert_allclose(c2w_out, c2w)
    assert c2w_out.dtype == np.float64

    bad = tmp_path / "r_1.npz"
    np.savez(bad, pose=c2w)  # missing intrinsics
    with pytest.raises(ValueError, match="intrinsics"):
        OmniObject3dDataset.read_omniobject3d_cam(str(bad))
    bad2 = tmp_path / "r_2.npz"
    np.savez(bad2, intrinsics=K[:2], pose=c2w)  # malformed K
    with pytest.raises(ValueError, match=r"\(3,3\)"):
        OmniObject3dDataset.read_omniobject3d_cam(str(bad2))


def test_intrinsics_assembly_override_and_error():
    K_npz = np.array([[1111.1111, 0, 400.0], [0, 1111.1111, 400.0], [0, 0, 1.0]])
    K = OmniObject3dDataset.omniobject3d_intrinsics(K_npz)
    assert K.shape == (3, 3) and K.dtype == np.float32 and K[2, 2] == 1.0
    np.testing.assert_allclose(K, K_npz)
    K2 = OmniObject3dDataset.omniobject3d_intrinsics(K_npz, override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0  # override wins
    with pytest.raises(ValueError, match="intrinsics"):
        OmniObject3dDataset.omniobject3d_intrinsics(None)
    with pytest.raises(ValueError, match="malformed"):
        OmniObject3dDataset.omniobject3d_intrinsics(np.zeros((3, 3)))  # fx=fy=0


def test_missing_dir_raises():
    with pytest.raises(ValueError, match="OMNIOBJECT3D_DIR"):
        OmniObject3dDataset(common_conf=_common_conf(), OMNIOBJECT3D_DIR=None)


# --- OmniObject3D integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_OO3D, reason=f"OmniObject3D data not found at {OMNIOBJECT3D_DIR}")
def test_oo3d_sample_schema_and_conventions():
    ds = OmniObject3dDataset(
        common_conf=_common_conf(),
        split="train",
        OMNIOBJECT3D_DIR=OMNIOBJECT3D_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.SKY_MASK not in ds.available_modalities    # object renders, no sky
    assert Modality.WORLD_POINTS not in ds.available_modalities

    n = 6
    batch = ds.get_data(seq_index=0, img_per_seq=n, aspect_ratio=1.0)
    v = batch["frame_num"]
    assert v == n
    assert batch["seq_name"] == "omniobject3d_" + SEQ
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
    assert (depth >= 0).all()                             # no sky encoding
    assert "sky_masks" not in batch                       # unadvertised -> key not emitted
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is False and batch["is_video"] is False
    assert "timestamps" not in batch                      # unordered views, none advertised

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_OO3D, reason=f"OmniObject3D data not found at {OMNIOBJECT3D_DIR}")
def test_lazy_construction_and_out_of_range_index():
    ds = OmniObject3dDataset(
        common_conf=_eval_common(),
        split="train",
        OMNIOBJECT3D_DIR=OMNIOBJECT3D_DIR,
        sequences=["anise"],          # bare pattern selects the whole category
        len_train=10,
    )
    assert ds.sequence_list_len > 1
    assert all(name.startswith("anise/") for name in ds.sequence_list)
    assert ds._frames_cache == {}     # no per-frame I/O at construction (lazy)
    with pytest.raises(ValueError, match="out of range"):
        ds.get_data(seq_index=ds.sequence_list_len, img_per_seq=2)
    # First access lists exactly that one sequence.
    assert ds.sequence_num_frames(0) == 100
    assert set(ds._frames_cache) == {ds.sequence_list[0]}


@pytest.mark.skipif(not HAVE_OO3D, reason=f"OmniObject3D data not found at {OMNIOBJECT3D_DIR}")
def test_oo3d_getitem_tuple_index():
    ds = OmniObject3dDataset(
        common_conf=_common_conf(),
        split="train",
        OMNIOBJECT3D_DIR=OMNIOBJECT3D_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_OO3D, reason=f"OmniObject3D data not found at {OMNIOBJECT3D_DIR}")
def test_oo3d_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _oo3d_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "sky_masks" not in sample               # unadvertised -> key not emitted
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_OO3D, reason=f"OmniObject3D data not found at {OMNIOBJECT3D_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _oo3d_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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
    # Order honored verbatim: each returned frame's pose (a per-frame quantity;
    # OmniObject3D views are all >50 deg apart, so poses are distinct) must equal
    # the pose obtained by fetching that single id alone, at the same position.
    for pos in (1, 3):
        single = vendor.get_data(
            seq_name=composed.sequence_name(0), ids=np.array([ids[pos]]), aspect_ratio=0.75
        )
        np.testing.assert_allclose(
            sample["extrinsics"][pos].numpy(), single["extrinsics"][0], atol=1e-6
        )
    np.testing.assert_array_equal(sample["ids"].numpy(), ids)


@pytest.mark.skipif(not HAVE_OO3D, reason=f"OmniObject3D data not found at {OMNIOBJECT3D_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [SEQ, SEQ2]
    composed = instantiate(
        _oo3d_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == 100   # r_0..r_99, survey-verified


@pytest.mark.skipif(not HAVE_OO3D, reason=f"OmniObject3D data not found at {OMNIOBJECT3D_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _oo3d_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (800, 800)                       # OmniObject3D native renders (H, W)

    composed.set_img_size(800)                        # native long side
    assert composed.img_size == 800
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (800, 800)

    composed.set_img_size(400)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (400, 400)


@pytest.mark.skipif(not HAVE_OO3D, reason=f"OmniObject3D data not found at {OMNIOBJECT3D_DIR}")
def test_reprojection_closure_locks_depth_pose_intrinsics():
    """World points from one view reprojected into a nearby view must land at
    depths matching that view's depth map: locks depth-scale x pose-convention x
    intrinsics consistency end-to-end through process_one_image. Uses the
    survey-verified adjacent-on-sphere pair (57 deg apart)."""
    ds = OmniObject3dDataset(
        common_conf=_eval_common(),
        split="train",
        OMNIOBJECT3D_DIR=OMNIOBJECT3D_DIR,
        sequences=[REPROJ_SEQ],
        len_train=10,
    )
    b = ds.get_data(
        seq_name=ds.sequence_list[0], ids=np.array(REPROJ_IDS), aspect_ratio=1.0
    )
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
