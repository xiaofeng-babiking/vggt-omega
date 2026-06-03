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
