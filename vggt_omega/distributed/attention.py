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
        # Only mask when shards are uneven (padding present). With equal-length
        # shards there is no padding, so we drop the mask -> SDPA can use the
        # FlashAttention kernel (~3x faster); a non-null attn_mask disables Flash.
        mask = None
        if any(length != max_len for length in lengths):
            mask = key_keep_mask(lengths, max_len, device=q.device)  # (1,1,1,world*max_len) bool
        return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask)


def _ring_exchange(send: torch.Tensor, recv_shape, cp_group, rank, world) -> torch.Tensor:
    """Send `send` to rank+1, receive the next block (shape `recv_shape`) from rank-1.

    Uses plain isend/irecv (portable across gloo and nccl). Non-blocking sends are
    posted alongside the matching receives, so the ring completes without deadlock.
    """
    send_to = (rank + 1) % world
    recv_from = (rank - 1) % world
    recv = send.new_empty(recv_shape)
    send_req = dist.isend(send.contiguous(), dst=send_to, group=cp_group)
    recv_req = dist.irecv(recv, src=recv_from, group=cp_group)
    recv_req.wait()
    send_req.wait()
    return recv


class RingAttention:
    """Blockwise online-softmax attention; rotate K,V around the ring. Exact, memory-optimal.

    Online-softmax stats are accumulated in fp32 (matching SDPA's internal
    accumulation) and cast back to the input dtype on return.
    """

    def __call__(self, q, k, v, cp_group) -> torch.Tensor:
        rank = dist.get_rank(cp_group)
        world = dist.get_world_size(cp_group)
        lengths = all_gather_ints(k.shape[2], cp_group, device=k.device)

        B, H, Lq, D = q.shape
        scale = D ** -0.5
        qf = q.float()
        m = torch.full((B, H, Lq, 1), float("-inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, Lq, 1), device=q.device, dtype=torch.float32)
        acc = torch.zeros((B, H, Lq, D), device=q.device, dtype=torch.float32)

        cur_k, cur_v = k.contiguous(), v.contiguous()
        for step in range(world):
            origin = (rank - step) % world
            if lengths[origin] > 0 and Lq > 0:
                s = torch.matmul(qf, cur_k.float().transpose(-1, -2)) * scale  # (B,H,Lq,Lk)
                m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
                corr = torch.exp(m - m_new)
                p = torch.exp(s - m_new)
                l = l * corr + p.sum(dim=-1, keepdim=True)
                acc = acc * corr + torch.matmul(p, cur_v.float())
                m = m_new
            if step < world - 1:
                nxt = (rank - step - 1) % world
                recv_shape = (B, H, lengths[nxt], D)
                cur_k = _ring_exchange(cur_k, recv_shape, cp_group, rank, world)
                cur_v = _ring_exchange(cur_v, recv_shape, cp_group, rank, world)

        if Lq == 0:
            return q
        out = acc / l.clamp_min(torch.finfo(torch.float32).tiny)
        return out.to(q.dtype)


def build_strategy(name: str) -> DistributedAttention:
    strategies = {"all_gather_kv": AllGatherKVAttention, "ring": RingAttention}
    if name not in strategies:
        raise ValueError(f"Unknown cp_strategy {name!r}; choices: {sorted(strategies)}")
    return strategies[name]()
