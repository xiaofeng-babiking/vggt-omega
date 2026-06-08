import torch
import torch.distributed as dist
import torch.nn.functional as F

from vggt_omega.distributed.attention import AllGatherKVAttention
from vggt_omega.distributed.tests._dist_test_util import run_distributed

# Fixed reference tensors (seed makes them deterministic across the test + workers).
B, H, D = 1, 4, 8
TOKENS = 5  # tokens per frame


def _make_qkv(num_frames):
    g = torch.Generator().manual_seed(1234)
    shape = (B, H, num_frames * TOKENS, D)
    return (torch.randn(shape, generator=g),
            torch.randn(shape, generator=g),
            torch.randn(shape, generator=g))


def _split_by_frame(t, counts):
    # t: (B,H,N*TOKENS,D) -> list of (B,H,count*TOKENS,D)
    out, start = [], 0
    for c in counts:
        out.append(t[:, :, start : start + c * TOKENS])
        start += c * TOKENS
    return out


def _worker(rank, world_size, num_frames, counts):
    q, k, v = _make_qkv(num_frames)
    qs = _split_by_frame(q, counts)[rank]
    ks = _split_by_frame(k, counts)[rank]
    vs = _split_by_frame(v, counts)[rank]
    return AllGatherKVAttention()(qs, ks, vs, dist.group.WORLD)


def _check(num_frames, world_size, counts):
    q, k, v = _make_qkv(num_frames)
    ref = F.scaled_dot_product_attention(q, k, v)
    parts = run_distributed(_worker, world_size, num_frames, counts)
    got = torch.cat(parts, dim=2)
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


def test_all_gather_kv_even():
    _check(num_frames=6, world_size=3, counts=[2, 2, 2])


def test_all_gather_kv_uneven():
    _check(num_frames=7, world_size=3, counts=[3, 2, 2])


def test_all_gather_kv_zero_frame_rank():
    _check(num_frames=2, world_size=3, counts=[1, 1, 0])
