import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.dynamic_replica import DynamicReplicaDataset

DYNAMIC_REPLICA_DIR = "/jfs/Data_4DFF/train_data/dynamic_replica_data"
HAVE_DR = os.path.isdir(DYNAMIC_REPLICA_DIR)
# One sequence dir (= 2 camera streams) keeps the integration tests fast.
SEQ = "009850-3_obj"


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


def _dr_dataset_cfg(seqs=(SEQ,), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.dynamic_replica.DynamicReplicaDataset",
                "split": "train",
                "DYNAMIC_REPLICA_DIR": DYNAMIC_REPLICA_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


def _median_reprojection_rel_err(world_src, pmask_src, E_dst, K_dst, depth_dst, min_points=500):
    """Median relative depth error of the src frame's valid world points
    reprojected into the dst frame (shared by the temporal and stereo closure
    tests). Asserts at least ``min_points`` points land on valid dst depth."""
    w = world_src[pmask_src]
    cam = w @ E_dst[:3, :3].T + E_dst[:3, 3]
    z = cam[:, 2]
    u = cam[:, 0] / z * K_dst[0, 0] + K_dst[0, 2]
    v = cam[:, 1] / z * K_dst[1, 1] + K_dst[1, 2]
    H, W = depth_dst.shape
    ok = (z > 0) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.round(u[ok]).astype(int)
    vi = np.round(v[ok]).astype(int)
    measured = depth_dst[vi, ui]
    valid = measured > 0
    assert valid.sum() >= min_points
    return float(np.median(np.abs(z[ok][valid] - measured[valid]) / measured[valid]))


# --- Dynamic-Replica-specific helper unit tests (no data required) ---


def test_pose_to_w2c_identity_and_translation():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    w2c = DynamicReplicaDataset.dynamic_replica_pose_to_w2c(c2w)
    assert w2c.shape == (3, 4) and w2c.dtype == np.float32
    np.testing.assert_allclose(w2c[:3, :3], np.eye(3), atol=1e-6)
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_pose_to_w2c_axis_remap_round_trip():
    # camera-to-world rotates cam axes 90 deg about world z: cam x -> world y.
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([0.5, -1.5, 2.0])
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = DynamicReplicaDataset.dynamic_replica_pose_to_w2c(c2w)
    # A point at the camera origin must land at the camera-frame origin...
    np.testing.assert_allclose(w2c[:3, :3] @ t + w2c[:, 3], 0.0, atol=1e-6)
    # ...and a point 1 m along the camera z axis must land at cam (0,0,1).
    p_world = t + R @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(w2c[:3, :3] @ p_world + w2c[:, 3], [0.0, 0.0, 1.0], atol=1e-6)


def test_pose_to_w2c_rejects_bad_input():
    with pytest.raises(ValueError, match="non-finite"):
        DynamicReplicaDataset.dynamic_replica_pose_to_w2c(np.full((4, 4), np.inf))
    with pytest.raises(ValueError, match=r"\(4,4\)"):
        DynamicReplicaDataset.dynamic_replica_pose_to_w2c(np.eye(3))


def test_depth_reader_units_and_invalid(tmp_path):
    arr = np.array([[-0.0, 1.5], [np.inf, np.nan], [-2.0, 3.25]], dtype=np.float32)
    p = tmp_path / "0.0.npy"
    np.save(p, arr)
    depth = DynamicReplicaDataset.read_dynamic_replica_depth(str(p))
    assert depth.dtype == np.float32
    # meters pass through unscaled; -0.0/inf/nan/negatives all map to 0 (no sky).
    np.testing.assert_allclose(depth, [[0.0, 1.5], [0.0, 0.0], [0.0, 3.25]])
    assert not (depth < 0).any() and not np.signbit(depth).any()


def test_camera_reader_pose_and_intrinsics(tmp_path):
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    K = np.array([[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]])
    p = tmp_path / "0.0.npz"
    np.savez(p, pose=c2w, intrinsics=K)
    w2c, K_read = DynamicReplicaDataset.read_dynamic_replica_camera(str(p))
    assert w2c.shape == (3, 4) and K_read.shape == (3, 3)
    assert K_read.dtype == np.float32
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)
    np.testing.assert_allclose(K_read, K)

    bad = tmp_path / "bad.npz"
    np.savez(bad, pose=c2w)  # missing 'intrinsics'
    with pytest.raises(ValueError, match="intrinsics"):
        DynamicReplicaDataset.read_dynamic_replica_camera(str(bad))


def test_intrinsics_default_and_override():
    K = DynamicReplicaDataset.dynamic_replica_intrinsics()
    assert K.shape == (3, 3) and K[2, 2] == 1.0
    assert K[0, 0] == 700.0 and K[1, 1] == 700.0
    assert K[0, 2] == 640.0 and K[1, 2] == 360.0
    K2 = DynamicReplicaDataset.dynamic_replica_intrinsics(override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0


def test_camera_id_for_stream():
    assert DynamicReplicaDataset.camera_id_for_stream("009850-3_obj/left") == 0
    assert DynamicReplicaDataset.camera_id_for_stream("009850-3_obj/right") == 1
    with pytest.raises(ValueError, match="camera"):
        DynamicReplicaDataset.camera_id_for_stream("009850-3_obj/center")


def test_sort_frame_stems_is_temporal_not_lexicographic():
    # The on-disk filename format, extended past the 10 s mark where
    # lexicographic order breaks ('10.0' < '9.96...' as strings).
    stems = [repr(i / 30.0) for i in range(330)]
    scrambled = sorted(stems)  # lexicographic really does scramble time
    assert scrambled != stems
    out = DynamicReplicaDataset.sort_frame_stems(scrambled)
    assert out == stems
    np.testing.assert_allclose([float(s) for s in out], np.arange(330) / 30.0)


def test_constructor_rejects_bad_args():
    with pytest.raises(ValueError, match="DYNAMIC_REPLICA_DIR"):
        DynamicReplicaDataset(common_conf=_common_conf(), DYNAMIC_REPLICA_DIR=None)
    with pytest.raises(ValueError, match="cameras"):
        DynamicReplicaDataset(
            common_conf=_common_conf(),
            DYNAMIC_REPLICA_DIR="/nonexistent",
            cameras=["center"],
        )


# --- Dynamic Replica integration tests (require the dataset on disk) ---


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_sample_schema_and_conventions():
    ds = DynamicReplicaDataset(
        common_conf=_common_conf(),
        split="train",
        DYNAMIC_REPLICA_DIR=DYNAMIC_REPLICA_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert Modality.EXTRINSICS in ds.available_modalities
    assert Modality.CAMERA_ID in ds.available_modalities
    # Indoor vendor contract: SKY_MASK advertised like TUM/7-Scenes (the
    # emitted masks are always all-False -- no sky in indoor synthetic data).
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

    assert img.shape[0] == v and img.ndim == 4            # (V,H,W,3) pre-permute
    assert depth.shape == img.shape[:1] + img.shape[1:3]  # (V,H,W)
    assert extr.shape == (v, 3, 4)
    assert intr.shape == (v, 3, 3)
    assert np.isfinite(extr).all()
    assert (depth[depth > 0]).size > 0                    # some valid metric depth
    assert not (depth < 0).any()                          # indoor: nothing encoded as sky
    sky = np.stack(batch["sky_masks"])
    assert sky.dtype == bool and sky.shape == depth.shape
    assert not sky.any()                                  # advertised but always all-False
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["camera_ids"].shape == (v,)
    assert batch["camera_ids"].dtype == np.int32
    assert set(np.unique(batch["camera_ids"])) <= {0, 1}
    assert batch["timestamps"].dtype == np.float64
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_camera_streams_ids_and_timestamps():
    """Each stereo camera is its own sequence; CAMERA_ID encodes left=0/right=1
    and timestamps come from the float-second filenames (= frame_idx / 30)."""
    ds = DynamicReplicaDataset(
        common_conf=_eval_common(),
        split="train",
        DYNAMIC_REPLICA_DIR=DYNAMIC_REPLICA_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    assert ds.sequence_list == [f"{SEQ}/left", f"{SEQ}/right"]

    left = ds.get_data(seq_name=f"{SEQ}/left", ids=np.array([0, 30]), aspect_ratio=1.0)
    right = ds.get_data(seq_name=f"{SEQ}/right", ids=np.array([0, 30]), aspect_ratio=1.0)
    assert left["seq_name"] == f"dynamic_replica_{SEQ}/left"
    np.testing.assert_array_equal(left["camera_ids"], [0, 0])
    np.testing.assert_array_equal(right["camera_ids"], [1, 1])
    # filename timestamps are frame_idx/30 s for the 30 fps export
    np.testing.assert_allclose(left["timestamps"], [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(right["timestamps"], [0.0, 1.0], atol=1e-9)
    # left/right share one world frame: same-instant camera centers differ by
    # the (per-sequence randomized 0.05-0.21 m) stereo baseline, not meters.
    EL, ER = left["extrinsics"][0], right["extrinsics"][0]
    cL = -EL[:3, :3].T @ EL[:, 3]
    cR = -ER[:3, :3].T @ ER[:, 3]
    assert 0.01 < np.linalg.norm(cL - cR) < 0.5


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_reprojection_closure_depth_pose_intrinsics():
    """World points from two NEARBY frames (dynamic scene: 2 frames = 1/15 s
    apart keeps object motion negligible) must be mutually consistent: frame A's
    valid world points reprojected into frame B land at depths matching B's
    depth map. This locks pose-convention x intrinsics through
    process_one_image. The 0.005 gate is tight enough to FAIL wrong conventions
    on this data (measured: correct c2w-OpenCV 0.0029 vs OpenGL axis flip
    0.0102) -- the flip the stereo closure below is blind to because the stereo
    baseline is pure-x. NOTE the closure alone is invariant to a uniform
    depth-scale error (z and measured depth scale together), so metric scale is
    pinned separately by the median-valid-depth bound."""
    ds = DynamicReplicaDataset(
        common_conf=_eval_common(),
        split="train",
        DYNAMIC_REPLICA_DIR=DYNAMIC_REPLICA_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    b = ds.get_data(seq_name=ds.sequence_list[0], ids=np.array([0, 2]), aspect_ratio=0.75)
    world = np.stack(b["world_points"])
    pmask = np.stack(b["point_masks"])
    extr = np.stack(b["extrinsics"])
    intr = np.stack(b["intrinsics"])
    depth = np.stack(b["depths"])

    # Metric scale: indoor Replica rooms, so valid depths must be O(meters)
    # (survey p50 ~1.9 m); a wrongly scaled (mm/cm/normalized) depth fails here.
    assert 0.5 < np.median(depth[depth > 0]) < 10.0

    rel = _median_reprojection_rel_err(world[0], pmask[0], extr[1], intr[1], depth[1])
    assert rel < 0.005


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_stereo_cross_camera_reprojection_closure():
    """SAME-INSTANT left[0] -> right[0] closure across the stereo pair (one
    shared world frame, identical dynamic-object positions, so the closure is
    exact even in a dynamic scene). The stereo baseline is large relative to
    the 1/15 s temporal step, so the 0.005 gate FAILS a w2c/c2w inversion
    error that the slow-moving temporal check cannot resolve (measured:
    correct 0.00054 vs pose-used-directly-as-w2c 0.0103)."""
    ds = DynamicReplicaDataset(
        common_conf=_eval_common(),
        split="train",
        DYNAMIC_REPLICA_DIR=DYNAMIC_REPLICA_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    left = ds.get_data(seq_name=f"{SEQ}/left", ids=np.array([0]), aspect_ratio=0.75)
    right = ds.get_data(seq_name=f"{SEQ}/right", ids=np.array([0]), aspect_ratio=0.75)
    rel = _median_reprojection_rel_err(
        left["world_points"][0],
        left["point_masks"][0],
        right["extrinsics"][0],
        right["intrinsics"][0],
        right["depths"][0],
    )
    assert rel < 0.005


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_getitem_tuple_index():
    ds = DynamicReplicaDataset(
        common_conf=_common_conf(),
        split="train",
        DYNAMIC_REPLICA_DIR=DYNAMIC_REPLICA_DIR,
        sequences=[SEQ],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _dr_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "timestamps" in sample                  # extended modalities carried through
    assert "camera_ids" in sample
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit UNORDERED ids (the training-identical inference path). The
    two must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _dr_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    ids = [0, 7, 3, 12]
    sample = composed.get_sample(0, ids=ids, aspect_ratio=0.75)

    assert sample["images"].shape[0] == len(ids)
    assert sample["images"].ndim == 4                       # (V, 3, H, W)
    assert 0.0 <= float(sample["images"].min()) and float(sample["images"].max()) <= 1.0
    assert sample["extrinsics"].shape == (len(ids), 3, 4)
    assert "modalities" in sample

    # Order honored verbatim: timestamps are frame_idx/30, a per-frame quantity.
    np.testing.assert_allclose(
        sample["timestamps"].numpy(), np.array(ids) / 30.0, atol=1e-9
    )

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
    np.testing.assert_allclose(sample["timestamps"].numpy(), batch["timestamps"])


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendor's real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences.
    Frame counts come from the lazy per-stream listing (300 frames per camera)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _dr_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    vendor = composed.base_dataset.datasets[0]
    assert composed.num_sequences() == len(vendor.sequence_list) == 2
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in vendor.sequence_list
        assert composed.sequence_num_frames(gi) == 300     # 10 s at 30 fps
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name])


@pytest.mark.skipif(not HAVE_DR, reason=f"Dynamic Replica data not found at {DYNAMIC_REPLICA_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _dr_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (720, 1280)                      # Dynamic Replica native (H, W)

    composed.set_img_size(1280)                       # native long side
    assert composed.img_size == 1280
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (720, 1280)

    composed.set_img_size(640)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (352, 640)   # int(640*9/16)=360 -> 352
