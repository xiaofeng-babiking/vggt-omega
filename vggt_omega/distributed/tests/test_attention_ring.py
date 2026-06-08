import torch
import torch.distributed as dist
import torch.nn.functional as F

from vggt_omega.distributed.attention import RingAttention
from vggt_omega.distributed.tests._dist_test_util import requires_dist, run_distributed

pytestmark = requires_dist  # gloo/CPU path disabled on torch 2.12.0+cu130 (RUN_DIST_TESTS=1 to run)

B, H, D = 1, 4, 8
TOKENS = 5


def _make_qkv(num_frames):
    g = torch.Generator().manual_seed(99)
    shape = (B, H, num_frames * TOKENS, D)
    return (torch.randn(shape, generator=g),
            torch.randn(shape, generator=g),
            torch.randn(shape, generator=g))


def _split_by_frame(t, counts):
    out, start = [], 0
    for c in counts:
        out.append(t[:, :, start : start + c * TOKENS])
        start += c * TOKENS
    return out


def _worker(rank, world_size, num_frames, counts):
    q, k, v = _make_qkv(num_frames)
    return RingAttention()(
        _split_by_frame(q, counts)[rank],
        _split_by_frame(k, counts)[rank],
        _split_by_frame(v, counts)[rank],
        dist.group.WORLD,
    )


def _check(num_frames, world_size, counts):
    q, k, v = _make_qkv(num_frames)
    ref = F.scaled_dot_product_attention(q, k, v)
    got = torch.cat(run_distributed(_worker, world_size, num_frames, counts), dim=2)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


def test_ring_even():
    _check(6, 3, [2, 2, 2])


def test_ring_uneven():
    _check(7, 3, [3, 2, 2])


def test_ring_zero_frame_rank():
    _check(2, 3, [1, 1, 0])


def test_ring_distinct_lengths():
    _check(6, 3, [3, 2, 1])
