"""Single-process unit tests for the ring block kernels and LSE merge (no process group)."""
import torch
import torch.nn.functional as F

from vggt_omega.distributed.attention import _block_attn_manual, _merge_block

B, H, D = 2, 3, 16
SCALE = D ** -0.5


def _rand(*shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def test_manual_block_matches_sdpa_and_lse():
    q, k, v = (_rand(B, H, n, D, seed=s) for n, s in ((11, 0), (23, 1), (23, 2)))
    out, lse = _block_attn_manual(q, k, v, SCALE)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-4)
    ref_lse = (torch.matmul(q, k.transpose(-1, -2)) * SCALE).logsumexp(dim=-1)
    torch.testing.assert_close(lse, ref_lse, atol=1e-5, rtol=1e-4)


def test_manual_block_chunked_equals_unchunked():
    q, k, v = (_rand(B, H, n, D, seed=s) for n, s in ((9, 3), (25, 4), (25, 5)))
    out_full, lse_full = _block_attn_manual(q, k, v, SCALE, key_chunk=10**9)
    out_chunk, lse_chunk = _block_attn_manual(q, k, v, SCALE, key_chunk=7)  # 25 % 7 != 0
    torch.testing.assert_close(out_chunk, out_full, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(lse_chunk, lse_full, atol=1e-6, rtol=1e-5)


def test_merge_blocks_equals_joint_attention():
    q, k, v = (_rand(B, H, n, D, seed=s) for n, s in ((9, 6), (30, 7), (30, 8)))
    out = torch.zeros(B, H, 9, D)
    lse = torch.full((B, H, 9), float("-inf"))  # initial accumulator state
    for blk in (slice(0, 13), slice(13, 30)):  # uneven split
        o, s = _block_attn_manual(q, k[:, :, blk], v[:, :, blk], SCALE)
        out, lse = _merge_block(out, lse, o, s)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-4)
