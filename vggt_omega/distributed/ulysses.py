"""Ulysses (DeepSpeed-style) all-to-all token<->head redistribution.

The aggregator's global attention is permutation-invariant (no positional
encoding), so we may freely redistribute tokens across GPUs. We shard the
sequence (frames) across the ``sp`` group everywhere *except* the attention
itself, where an all-to-all swaps token-sharding for head-sharding: each GPU
then holds all tokens but only ``H / sp_size`` heads and runs an ordinary,
exact local SDPA. Inference only -> no autograd, so these are plain functional
collectives.

Layout convention: tensors are ``(B, H, N, d)`` (batch, heads, tokens,
head_dim), matching ``SelfAttention.compute_attention`` after its
``transpose(1, 2)``.
"""
from __future__ import annotations

import torch
import torch.distributed as dist


def scatter_heads(x: torch.Tensor, group: dist.ProcessGroup | None, world_size: int) -> torch.Tensor:
    """``(B, H, N_local, d)`` -> ``(B, H // G, G * N_local, d)``.

    Scatters the head dim across ranks, gathers the token dim. After this call
    every rank holds *all* tokens but only its ``H // G`` heads.
    """
    B, H, N_local, d = x.shape
    if H % world_size != 0:
        raise ValueError(f"num_heads ({H}) must be divisible by sp world_size ({world_size})")
    heads_local = H // world_size
    # (B, G, Hg, N_local, d) -> (G, B, Hg, N_local, d); dim0 is the scatter axis.
    x = x.reshape(B, world_size, heads_local, N_local, d).permute(1, 0, 2, 3, 4).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    # out dim0 now indexes the *source* rank == a token block.
    out = out.permute(1, 2, 0, 3, 4).contiguous()  # (B, Hg, G, N_local, d)
    return out.reshape(B, heads_local, world_size * N_local, d)


def gather_heads(x: torch.Tensor, group: dist.ProcessGroup | None, world_size: int) -> torch.Tensor:
    """Inverse of :func:`scatter_heads`: ``(B, H // G, N_full, d)`` -> ``(B, H, N_local, d)``."""
    B, heads_local, N_full, d = x.shape
    if N_full % world_size != 0:
        raise ValueError(f"N_full ({N_full}) must be divisible by sp world_size ({world_size})")
    N_local = N_full // world_size
    # (B, Hg, G, N_local, d) -> (G, B, Hg, N_local, d); dim0 is the scatter axis.
    x = x.reshape(B, heads_local, world_size, N_local, d).permute(2, 0, 1, 3, 4).contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    # out dim0 now indexes the *source* rank == a head block.
    out = out.permute(1, 0, 2, 3, 4).contiguous()  # (B, G, Hg, N_local, d)
    return out.reshape(B, world_size * heads_local, N_local, d)
