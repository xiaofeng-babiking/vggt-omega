import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.modality import Modality, validate_sample
from vggt_omega.datasets.vendors.tum import TumDataset

TUM_DIR = "/jfs/guibiao/streamVGGT/data/eval/tum"
HAVE_TUM = os.path.isdir(TUM_DIR)


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


def _tum_dataset_cfg(seqs=("rgbd_dataset_freiburg3_sitting_halfsphere",), n=20):
    return {
        "_target_": "vggt_omega.datasets.composed_dataset.ComposedDataset",
        "dataset_configs": [
            {
                "_target_": "vggt_omega.datasets.vendors.tum.TumDataset",
                "split": "train",
                "TUM_DIR": TUM_DIR,
                "sequences": list(seqs),
                "len_train": n,
            }
        ],
    }


# --- TUM-specific helper unit tests (no data required) ---


def test_tum_pose_to_w2c_inverts_c2w():
    w2c = TumDataset.tum_pose_to_w2c(np.zeros(3), (0.0, 0.0, 0.0, 1.0))
    assert w2c.shape == (3, 4)
    np.testing.assert_allclose(w2c, np.hstack([np.eye(3), np.zeros((3, 1))]), atol=1e-6)


def test_tum_pose_to_w2c_translation():
    w2c = TumDataset.tum_pose_to_w2c(np.array([1.0, 2.0, 3.0]), (0.0, 0.0, 0.0, 1.0))
    np.testing.assert_allclose(w2c[:, 3], [-1.0, -2.0, -3.0], atol=1e-6)


def test_tum_intrinsics_fr3_and_override_and_unknown():
    K = TumDataset.tum_intrinsics("rgbd_dataset_freiburg3_sitting_halfsphere")
    assert K.shape == (3, 3) and K[0, 0] > 0 and K[2, 2] == 1.0
    K2 = TumDataset.tum_intrinsics("anything", override=[100.0, 100.0, 50.0, 50.0])
    assert K2[0, 0] == 100.0 and K2[0, 2] == 50.0
    with pytest.raises(ValueError, match="intrinsics"):
        TumDataset.tum_intrinsics("no_camera_here")


# --- TUM integration tests (require the TUM dataset on disk) ---


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_tum_sample_schema_and_conventions():
    ds = TumDataset(
        common_conf=_common_conf(),
        split="train",
        TUM_DIR=TUM_DIR,
        sequences=["rgbd_dataset_freiburg3_sitting_halfsphere"],
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
    assert np.isfinite(world[pmask]).all()                # valid points are finite
    assert batch["is_metric"] is True and batch["is_video"] is True

    validate_sample(batch, ds.available_modalities)


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_tum_getitem_tuple_index():
    ds = TumDataset(
        common_conf=_common_conf(),
        split="train",
        TUM_DIR=TUM_DIR,
        sequences=["rgbd_dataset_freiburg3_sitting_halfsphere"],
        len_train=10,
    )
    batch = ds[(0, 4, 1.0)]   # (seq_index, img_per_seq, aspect_ratio)
    assert batch["frame_num"] == 4


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_tum_through_composed_dataset():
    """ComposedDataset tensorizes + carries extended modalities (no DDP needed)."""
    from hydra.utils import instantiate

    composed = instantiate(
        _tum_dataset_cfg(), common_config=_integration_common(), _recursive_=False
    )
    sample = composed[(0, 4, 1.0)]                 # (seq_idx, img_per_seq, aspect)
    assert sample["images"].ndim == 4              # (V, 3, H, W)
    assert sample["images"].shape[0] == 4
    assert "timestamps" in sample                  # extended modality carried through
    assert "modalities" in sample
    assert sample["extrinsics"].shape == (4, 3, 4)


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_tum_full_dynamic_loader():
    """Full DynamicTorchDataset path. DynamicDistributedSampler subclasses
    torch DistributedSampler, which requires an initialized process group, so
    spin up a 1-process gloo group for the test (num_workers=0 avoids the
    RANK-env worker_init path)."""
    import torch.distributed as dist
    from hydra.utils import instantiate

    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    try:
        # inside_random=True so TupleConcatDataset and TumDataset both ignore
        # the raw sampler index (0..len_train-1) and pick randomly from the
        # actual sequence list — necessary when len_train >> sequence_list_len.
        loader_common = OmegaConf.merge(
            _integration_common(), OmegaConf.create({"inside_random": True})
        )
        loader_obj = instantiate(
            {
                "_target_": "vggt_omega.datasets.dynamic_dataloader.DynamicTorchDataset",
                "num_workers": 0,
                "shuffle": False,
                "pin_memory": False,
                "max_img_per_gpu": 12,
            },
            dataset=OmegaConf.create(_tum_dataset_cfg()),
            common_config=loader_common,
            _recursive_=False,
        )
        batch = next(iter(loader_obj.get_loader(epoch=0)))
        assert batch["images"].ndim == 5           # (B, V, 3, H, W)
        assert "timestamps" in batch
    finally:
        if created:
            dist.destroy_process_group()


# --- ComposedDataset explicit-ids eval path (drives inference.py) ---


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_composed_get_sample_honors_explicit_ids_and_tensorizes():
    """ComposedDataset.get_sample tensorizes EXACTLY like get_data + the __getitem__
    block, for explicit ordered ids (the training-identical inference path). The two
    must not drift, and the requested id order must be honored verbatim."""
    import torch
    from hydra.utils import instantiate

    composed = instantiate(
        _tum_dataset_cfg(), common_config=_eval_common(), _recursive_=False
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
    # Order honored: per-frame timestamps follow the requested id order.
    np.testing.assert_allclose(sample["timestamps"].numpy(), batch["timestamps"])


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_composed_sequence_enumeration():
    """num_sequences / sequence_name / sequence_num_frames expose the vendors' real
    sequence_list (not the virtual len_train), so inference.py can iterate sequences."""
    from hydra.utils import instantiate

    seqs = [
        "rgbd_dataset_freiburg3_sitting_halfsphere",
        "rgbd_dataset_freiburg3_sitting_xyz",
    ]
    composed = instantiate(
        _tum_dataset_cfg(seqs=seqs), common_config=_eval_common(), _recursive_=False
    )
    assert composed.num_sequences() == 2
    vendor = composed.base_dataset.datasets[0]
    for gi in range(composed.num_sequences()):
        name = composed.sequence_name(gi)
        assert name in seqs
        assert composed.sequence_num_frames(gi) == len(vendor.data_store[name])


@pytest.mark.skipif(not HAVE_TUM, reason=f"TUM data not found at {TUM_DIR}")
def test_composed_native_geometry_and_set_img_size():
    """ComposedDataset reads native frame geometry from the data, and set_img_size
    drives the get_sample target resolution -- so inference can source img_size /
    aspect from the dataset instead of hardcoded constants."""
    from hydra.utils import instantiate

    composed = instantiate(
        _tum_dataset_cfg(), common_config=_eval_common(), _recursive_=False
    )
    h, w = composed.native_image_size(0)
    assert (h, w) == (480, 640)                       # TUM native VGA (H, W)

    composed.set_img_size(640)                        # native long side
    assert composed.img_size == 640
    s = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s["images"].shape[-2:]) == (480, 640)

    composed.set_img_size(320)                        # half-res long side, /16 snapped
    s2 = composed.get_sample(0, ids=[0, 1], aspect_ratio=h / w)
    assert tuple(s2["images"].shape[-2:]) == (240, 320)
