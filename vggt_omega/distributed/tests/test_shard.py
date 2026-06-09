import numpy as np
import torch

from vggt_omega.distributed.shard import (
    frame_counts_for,
    key_keep_mask,
    pad_seq_to,
    shard_frame_ids,
)


def test_shard_even():
    ids = np.arange(6)
    assert shard_frame_ids(ids, 0, 3).tolist() == [0, 1]
    assert shard_frame_ids(ids, 1, 3).tolist() == [2, 3]
    assert shard_frame_ids(ids, 2, 3).tolist() == [4, 5]


def test_shard_uneven_remainder_goes_to_low_ranks():
    ids = np.arange(7)
    assert shard_frame_ids(ids, 0, 3).tolist() == [0, 1, 2]  # base+1
    assert shard_frame_ids(ids, 1, 3).tolist() == [3, 4]
    assert shard_frame_ids(ids, 2, 3).tolist() == [5, 6]
    assert frame_counts_for(7, 3) == [3, 2, 2]


def test_shard_fewer_frames_than_ranks_gives_empty():
    ids = np.arange(2)
    assert shard_frame_ids(ids, 0, 4).tolist() == [0]
    assert shard_frame_ids(ids, 2, 4).tolist() == []
    assert frame_counts_for(2, 4) == [1, 1, 0, 0]


def test_pad_seq_to_pads_with_zeros_on_seq_dim():
    x = torch.ones(1, 2, 3, 4)  # (B,H,seq=3,D)
    out = pad_seq_to(x, 5, dim=2)
    assert out.shape == (1, 2, 5, 4)
    assert out[:, :, :3].eq(1).all() and out[:, :, 3:].eq(0).all()


def test_key_keep_mask_marks_only_real_keys():
    mask = key_keep_mask([2, 1], max_len=2, device="cpu")  # 2 ranks
    assert mask.shape == (1, 1, 1, 4)
    assert mask.flatten().tolist() == [True, True, True, False]
