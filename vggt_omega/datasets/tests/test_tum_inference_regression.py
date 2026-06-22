"""Regression test: run VGGT-Omega inference on a TUM sequence with arbitrary frames.

Exercises the full BaseSequence -> SequenceDataset -> ComposedDataset -> model
path that inference.py uses, proving the new BaseSequence-backed vendors feed the
model exactly like the training loader.

Two tiers:
  * data-pipeline tier (always runs when the TUM data is present): builds the
    dataset from the shipped tum.yaml and tensorizes an ARBITRARY set of frame
    ids through ComposedDataset.get_sample, asserting the training contract
    (shapes / dtypes / value ranges).
  * model tier (runs only when CUDA + the checkpoint are present): additionally
    runs the model forward on those frames and checks the prediction shapes.
"""
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from vggt_omega.datasets.composed_dataset import ComposedDataset

_CFG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "config", "tum.yaml"
)
_TUM_DIR = "/jfs/guibiao/streamVGGT/data/eval/tum"
_CKPT = "/jfs/jing.feng/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt"

HAVE_TUM = os.path.isdir(_TUM_DIR)
HAVE_MODEL = torch.cuda.is_available() and os.path.isfile(_CKPT)

# Arbitrary, non-contiguous, non-uniform frame ids (the "arbitrary frames" case).
ARBITRARY_IDS = [0, 7, 23, 58, 131]


def _build_dataset() -> ComposedDataset:
    cfg = OmegaConf.load(_CFG)
    return ComposedDataset(
        dataset_configs=cfg.dataset.dataset_configs, common_config=cfg.common_config
    )


@pytest.fixture(scope="module")
def dataset():
    return _build_dataset()


# --------------------------------------------------------------------------- #
# data pipeline tier
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_TUM, reason="TUM dataset not available")
def test_dataset_enumeration(dataset):
    assert dataset.num_sequences() >= 1
    n = dataset.sequence_num_frames(0)
    assert n > max(ARBITRARY_IDS)
    h, w = dataset.native_image_size(0)
    assert (h, w) == (480, 640)


@pytest.mark.skipif(not HAVE_TUM, reason="TUM dataset not available")
def test_arbitrary_frames_tensorize(dataset):
    s = dataset.get_sample(0, ids=ARBITRARY_IDS, aspect_ratio=1.0, num_workers=1)
    v = len(ARBITRARY_IDS)
    # training-identical contract
    assert s["images"].shape[0] == v and s["images"].shape[1] == 3
    assert s["images"].dtype == torch.get_default_dtype()
    assert 0.0 <= float(s["images"].min()) and float(s["images"].max()) <= 1.0
    assert tuple(s["extrinsics"].shape) == (v, 3, 4)
    assert tuple(s["intrinsics"].shape) == (v, 3, 3)
    assert s["depths"].shape[0] == v
    assert s["ids"].tolist() == ARBITRARY_IDS
    # images are square at the configured img_size
    assert s["images"].shape[-1] == s["images"].shape[-2]


@pytest.mark.skipif(not HAVE_TUM, reason="TUM dataset not available")
def test_se3_sampled_frames(dataset):
    # img_per_seq path -> SE(3) arc-length sampled ids over the (eval) full window.
    vendor = dataset.base_dataset.datasets[0]
    batch = vendor.get_data(seq_index=0, img_per_seq=8)
    ids = np.asarray(batch["ids"])
    assert len(ids) == 8
    assert ids[0] == 0 and ids[-1] == dataset.sequence_num_frames(0) - 1
    assert np.all(np.diff(ids) > 0)  # strictly increasing, deduped


# --------------------------------------------------------------------------- #
# model tier (CUDA + checkpoint)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (HAVE_TUM and HAVE_MODEL), reason="CUDA + checkpoint required")
def test_inference_arbitrary_frames(dataset):
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.pose_enc import encoding_to_camera

    sample = dataset.get_sample(0, ids=ARBITRARY_IDS, aspect_ratio=1.0, num_workers=1)
    images = sample["images"].contiguous().cuda()

    model = VGGTOmega().cuda().eval()
    model.load_state_dict(torch.load(_CKPT, map_location="cpu"))
    with torch.inference_mode():
        pred = model(images)

    v, _, h, w = images.shape
    extr, intr = encoding_to_camera(pred["pose_enc"], pred["images"].shape[-2:])
    assert extr.shape[1] == v and tuple(extr.shape[-2:]) == (3, 4)
    assert intr.shape[1] == v and tuple(intr.shape[-2:]) == (3, 3)
    depth = pred["depth"].float().cpu().numpy()[0]
    assert depth.shape[0] == v
    assert np.isfinite(depth).any()
