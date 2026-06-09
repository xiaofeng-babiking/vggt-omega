"""Run a SelfAttentionBlock's eval forward with its global attention distributed.

Reuses the block's own weights (norm/qkv/qk-norm/proj/layerscale/mlp) and swaps
ONLY the SDPA core for a DistributedAttention strategy. Mirrors the eval branch
of SelfAttentionBlock._forward_list (models/layers/block.py:195-201) and
SelfAttention.compute_attention (models/layers/attention.py:123-138).
"""
import torch


def distributed_block_forward(block, x_local, cp_group, strategy, rope=None):
    """x_local: (B, seq_local, C). Returns (B, seq_local, C)."""
    attn = block.attn
    B, N, C = x_local.shape
    heads = attn.num_heads
    head_dim = C // heads

    y = block.norm1(x_local)
    qkv = attn.qkv(y).reshape(B, N, 3, heads, head_dim)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, heads, N, head_dim)
    if attn.use_qk_norm:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    if rope is not None:
        q, k = attn.apply_rope(q, k, rope)

    o = strategy(q, k, v, cp_group)  # (B, heads, N, head_dim) -- the only communication
    o = o.transpose(1, 2).reshape(B, N, C)
    o = attn.proj(o)

    x_attn = x_local + block.ls1(o)
    return x_attn + block.ls2(block.mlp(block.norm2(x_attn)))
