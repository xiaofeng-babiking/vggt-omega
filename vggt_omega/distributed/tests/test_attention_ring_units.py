"""Single-process unit tests for the ring block kernels and LSE merge (no process group)."""
import pytest
import torch
import torch.nn.functional as F

from vggt_omega.distributed.attention import _block_attn_flash, _block_attn_manual, _merge_block

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash kernel is CUDA-only")
def test_flash_block_matches_manual_on_gpu():
    """Pins the private aten flash op's contract: (out, lse) tuple layout, lse
    semantics (natural-log, includes scale, shape (B,H,Lq)). A torch upgrade
    changing the op would otherwise surface only in manual GPU validation."""
    q, k, v = (_rand(B, H, n, D, seed=s).to("cuda", torch.bfloat16) for n, s in ((11, 9), (23, 10), (23, 11)))
    out_f, lse_f = _block_attn_flash(q, k, v, SCALE)
    out_m, lse_m = _block_attn_manual(q, k, v, SCALE)
    assert out_f.dtype == torch.bfloat16 and lse_f.dtype == torch.float32
    torch.testing.assert_close(out_f.float(), out_m, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse_f, lse_m, atol=2e-2, rtol=1e-3)


def test_merge_blocks_equals_joint_attention():
    q, k, v = (_rand(B, H, n, D, seed=s) for n, s in ((9, 6), (30, 7), (30, 8)))
    out = torch.zeros(B, H, 9, D)
    lse = torch.full((B, H, 9), float("-inf"))  # initial accumulator state
    for blk in (slice(0, 13), slice(13, 30)):  # uneven split
        o, s = _block_attn_manual(q, k[:, :, blk], v[:, :, blk], SCALE)
        out, lse = _merge_block(out, lse, o, s)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5, rtol=1e-4)
