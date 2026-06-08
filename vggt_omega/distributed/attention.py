"""Pluggable distributed (frame-sharded) global self-attention strategies.

Each strategy takes per-rank q/k/v of shape (B, heads, seq_local, head_dim) and
returns the per-rank attention output (B, heads, seq_local, head_dim), computing
attention over the *global* (all-rank) key/value sequence. All strategies are
mathematically equivalent to single-rank SDPA over the concatenated sequence;
only fp reduction order differs.
"""
from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .process_group import all_gather_ints
from .shard import key_keep_mask, pad_seq_to


class DistributedAttention(Protocol):
    def __call__(self, q, k, v, cp_group) -> torch.Tensor: ...


def _all_gather_concat_seq(x_padded: torch.Tensor, cp_group) -> torch.Tensor:
    """All-gather equal-shaped (B,H,max_len,D) tensors and concat along seq -> (B,H,world*max_len,D)."""
    world = dist.get_world_size(cp_group)
    out = [torch.empty_like(x_padded) for _ in range(world)]
    dist.all_gather(out, x_padded.contiguous(), group=cp_group)
    return torch.cat(out, dim=2)


class AllGatherKVAttention:
    """Gather all K,V to every rank, then local-Q vs global-KV SDPA. Exact."""

    def __call__(self, q, k, v, cp_group) -> torch.Tensor:
        lengths = all_gather_ints(k.shape[2], cp_group, device=k.device)
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            return q  # everything empty; preserves shape (B,H,0,D)
        k_full = _all_gather_concat_seq(pad_seq_to(k, max_len, dim=2), cp_group)
        v_full = _all_gather_concat_seq(pad_seq_to(v, max_len, dim=2), cp_group)
        mask = key_keep_mask(lengths, max_len, device=q.device)  # (1,1,1,world*max_len) bool
        return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)


def build_strategy(name: str) -> DistributedAttention:
    strategies = {"all_gather_kv": AllGatherKVAttention}  # "ring" added in a later task
    if name not in strategies:
        raise ValueError(f"Unknown cp_strategy {name!r}; choices: {sorted(strategies)}")
    return strategies[name]()
