"""slice_expand_and_flatten applies the first-frame token only on the first shard."""
from __future__ import annotations

import torch

from vggt_omega.models.aggregator import slice_expand_and_flatten


def _token() -> torch.Tensor:
    # (1, 2, 1, D): index 0 = first-frame token, index 1 = other-frames token.
    t = torch.zeros(1, 2, 1, 4)
    t[:, 0] = 1.0  # first-frame token = all ones
    t[:, 1] = 2.0  # other-frames token = all twos
    return t


def test_first_shard_uses_first_frame_token_at_index_0():
    out = slice_expand_and_flatten(_token(), batch_size=1, num_frames=3, is_first_shard=True)
    # (3, 1, 4): frame 0 -> ones, frames 1,2 -> twos
    assert torch.equal(out[0], torch.ones(1, 4))
    assert torch.equal(out[1], torch.full((1, 4), 2.0))
    assert torch.equal(out[2], torch.full((1, 4), 2.0))


def test_non_first_shard_uses_other_token_everywhere():
    out = slice_expand_and_flatten(_token(), batch_size=1, num_frames=3, is_first_shard=False)
    assert torch.equal(out, torch.full((3, 1, 4), 2.0))


def test_default_is_first_shard_true_preserves_legacy_behavior():
    out = slice_expand_and_flatten(_token(), batch_size=1, num_frames=2)
    assert torch.equal(out[0], torch.ones(1, 4))
    assert torch.equal(out[1], torch.full((1, 4), 2.0))
