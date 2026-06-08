import torch
import torch.distributed as dist

from vggt_omega.distributed.attention import AllGatherKVAttention
from vggt_omega.distributed.block import distributed_block_forward
from vggt_omega.distributed.tests._dist_test_util import run_distributed
from vggt_omega.models.layers import Mlp
from vggt_omega.models.layers.block import SelfAttentionBlock

DIM, HEADS, TOKENS = 32, 4, 5


def _make_block():
    torch.manual_seed(7)
    block = SelfAttentionBlock(
        dim=DIM, num_heads=HEADS, ffn_ratio=4.0, qkv_bias=True, proj_bias=True,
        ffn_bias=True, ffn_layer=Mlp, init_values=1e-5, use_qk_norm=True, mask_k_bias=True,
    ).eval()
    # mask_k_bias buffers default to NaN; set to ones so the bias is active and finite.
    for m in block.modules():
        if hasattr(m, "bias_mask"):
            torch.nn.init.ones_(m.bias_mask)
    return block


def _make_x(num_frames):
    g = torch.Generator().manual_seed(3)
    return torch.randn(1, num_frames * TOKENS, DIM, generator=g)


def _split_by_frame(x, counts):
    out, start = [], 0
    for c in counts:
        out.append(x[:, start : start + c * TOKENS])
        start += c * TOKENS
    return out


def _worker(rank, world_size, state_dict, num_frames, counts):
    block = _make_block()
    block.load_state_dict(state_dict)
    x_local = _split_by_frame(_make_x(num_frames), counts)[rank]
    with torch.no_grad():
        return distributed_block_forward(
            block, x_local, dist.group.WORLD, AllGatherKVAttention(), rope=None
        )


def test_block_g2_matches_single_rank_block():
    block = _make_block()
    x = _make_x(6)
    with torch.no_grad():
        ref = block(x, None)  # single-rank full-sequence block
    parts = run_distributed(_worker, 3, block.state_dict(), 6, [2, 2, 2])
    got = torch.cat(parts, dim=1)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_block_g2_uneven():
    block = _make_block()
    x = _make_x(7)
    with torch.no_grad():
        ref = block(x, None)
    parts = run_distributed(_worker, 3, block.state_dict(), 7, [3, 2, 2])
    got = torch.cat(parts, dim=1)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
