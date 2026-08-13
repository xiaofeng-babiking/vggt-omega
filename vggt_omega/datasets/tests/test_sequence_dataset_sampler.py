"""Tests for the config-selectable frame sampler on SequenceDataset
(frame_sampler = "se3" | "random" | "se3_random")."""
import random
import tempfile

import numpy as np
import pytest
from omegaconf import OmegaConf

from vggt_omega.datasets.dataloaders.sequence_dataset import (
    _FRAME_SAMPLERS,
    SequenceDataset,
)
from vggt_omega.datasets.samplers import (
    sample_frame_indices,
    sample_se3_random,
    sample_se3_trajectory,
)
from vggt_omega.datasets.tests.test_se3_sampler import (
    _PoseSeq,
    _rng,
    _still_then_move_rows,
    _traj,
)

_N = 8


def _conf():
    return OmegaConf.create(
        {
            "img_size": 224,
            "patch_size": 14,
            "augs": {"scales": None, "aspects": [1.0, 1.0]},
            "rescale": True,
            "rescale_aug": False,
            "landscape_check": False,
            "training": True,
            "inside_random": True,
            "allow_duplicate_img": False,
            "get_nearby": False,
            "samples_per_epoch": 100,
        }
    )


def _make_dataset(**kw):
    from vggt_omega.datasets.tests.test_sequences import _write_tum_sequence
    from vggt_omega.datasets.sequences.tum import TumSequence

    root = tempfile.mkdtemp()
    _write_tum_sequence(root, "rgbd_dataset_freiburg1_xyz", n=_N)
    return SequenceDataset(_conf(), TumSequence, root, **kw)


def test_sampler_registry_is_complete():
    assert _FRAME_SAMPLERS == {
        "se3": sample_se3_trajectory,
        "random": sample_frame_indices,
        "se3_random": sample_se3_random,
    }


def test_default_is_se3():
    assert _make_dataset().frame_sampler == "se3"


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="frame_sampler"):
        _make_dataset(frame_sampler="bogus")


@pytest.mark.parametrize("name", ["se3", "random", "se3_random"])
def test_sample_ids_contract_for_every_sampler(name):
    # Every sampler choice must yield exactly num distinct, sorted, in-range
    # ids through the dataset's random-window path (the "se3" top-up
    # shortfall case is exercised in test_top_up_fills_se3_dedup_shortfall).
    ds = _make_dataset(frame_sampler=name)
    seq = ds._sequence(ds.sequence_list[0])
    sensor = ds._sensor(seq)
    for seed in range(10):
        random.seed(seed)      # window draw
        np.random.seed(seed)   # frame draw
        ids = ds._sample_ids(seq, sensor, num=4)
        ids = [int(i) for i in ids]
        assert len(ids) == 4 and len(set(ids)) == 4
        assert ids == sorted(ids)
        assert all(0 <= i < _N for i in ids)


def test_num_capped_by_sequence_length():
    ds = _make_dataset(frame_sampler="se3_random")
    seq = ds._sequence(ds.sequence_list[0])
    sensor = ds._sensor(seq)
    ids = ds._sample_ids(seq, sensor, num=_N + 5)
    assert list(ids) == list(range(_N))


def test_top_up_fills_se3_dedup_shortfall():
    # A long static head makes the equal-arc "se3" sampler dedup below num
    # (asserted as a precondition); _sample_ids must top the result up to
    # exactly num distinct, sorted, in-window ids that still contain every
    # frame the sampler picked.
    pose_seq = _PoseSeq(_traj(_still_then_move_rows(_rng(0), 8, 2)))  # 10 frames
    raw, _ = sample_se3_trajectory(pose_seq, 0, num=8, start=0, end=9)
    assert len(raw) < 8  # precondition: dedup actually shrank the se3 result

    ds = _make_dataset(frame_sampler="se3")
    ds.inside_random = False  # full-window path: the same [0, 9] window
    ids = [int(i) for i in ds._sample_ids(pose_seq, 0, num=8)]
    assert len(ids) == 8 and len(set(ids)) == 8
    assert ids == sorted(ids)
    assert all(0 <= i <= 9 for i in ids)
    assert set(raw).issubset(ids)  # top-up preserved the sampler's picks
