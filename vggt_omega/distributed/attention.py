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

from .process_group import all_gather_ints, p2p_group_for
from .shard import pad_seq_to


class DistributedAttention(Protocol):
    def __call__(self, q, k, v, cp_group) -> torch.Tensor: ...


def _all_gather_blocks(x_padded: torch.Tensor, cp_group) -> list[torch.Tensor]:
    """All-gather equal-shaped (B,H,max_len,D) tensors -> list of the `world` rank blocks."""
    world = dist.get_world_size(cp_group)
    out = [torch.empty_like(x_padded) for _ in range(world)]
    dist.all_gather(out, x_padded.contiguous(), group=cp_group)
    return out


class AllGatherKVAttention:
    """Gather all K,V to every rank, then local-Q vs global-KV SDPA. Exact."""

    def __call__(self, q, k, v, cp_group) -> torch.Tensor:
        lengths = all_gather_ints(k.shape[2], cp_group, device=k.device)
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            return q  # everything empty; preserves shape (B,H,0,D)
        # Variable-length all-gather, emulated as pad -> all_gather -> TRIM:
        # all_gather needs equal shapes, so we pad each rank's K/V to max_len,
        # gather, then drop each rank's padding and concatenate into the real,
        # contiguous global K/V. Because no padded keys remain we pass NO attn_mask,
        # so SDPA uses FlashAttention even when shards are uneven (a non-null mask
        # forces the ~3x-slower memory-efficient kernel). Exact: global attention
        # has no cross-frame positional encoding, so this equals masked-padded SDPA.
        k_blocks = _all_gather_blocks(pad_seq_to(k, max_len, dim=2), cp_group)
        v_blocks = _all_gather_blocks(pad_seq_to(v, max_len, dim=2), cp_group)
        k_full = torch.cat([blk[:, :, :n] for blk, n in zip(k_blocks, lengths)], dim=2)
        v_full = torch.cat([blk[:, :, :n] for blk, n in zip(v_blocks, lengths)], dim=2)
        return F.scaled_dot_product_attention(q, k_full, v_full)


_MANUAL_KEY_CHUNK = 4096  # keys per online-softmax chunk on the manual path


def _block_attn_manual(q, k, v, scale, key_chunk=_MANUAL_KEY_CHUNK):
    """Exact attention of q over ONE K/V block via chunked online softmax.

    Returns (out, lse), both fp32: out (B,H,Lq,D) is the normalized attention
    output, lse (B,H,Lq) = log sum_j exp(scale * q.k_j). Iterating keys in
    chunks keeps memory O(Lq * key_chunk) instead of O(Lq * Lk). Serves CPU
    (gloo tests) and fp32-on-GPU (camera head), where the flash kernel is
    unavailable.
    """
    qf = q.float()
    B, H, Lq, Dh = qf.shape
    m = torch.full((B, H, Lq, 1), float("-inf"), device=q.device, dtype=torch.float32)
    l = torch.zeros((B, H, Lq, 1), device=q.device, dtype=torch.float32)
    acc = torch.zeros((B, H, Lq, Dh), device=q.device, dtype=torch.float32)
    for start in range(0, k.shape[2], key_chunk):
        kc = k[:, :, start : start + key_chunk].float()
        vc = v[:, :, start : start + key_chunk].float()
        s = torch.matmul(qf, kc.transpose(-1, -2)) * scale
        m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
        corr = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        l = l * corr + p.sum(dim=-1, keepdim=True)
        acc = acc * corr + torch.matmul(p, vc)
        m = m_new
    tiny = torch.finfo(torch.float32).tiny
    out = acc / l.clamp_min(tiny)
    lse = (m + l.clamp_min(tiny).log()).squeeze(-1)
    return out, lse


def _block_attn_flash(q, k, v, scale):
    """One K/V block via the FlashAttention kernel. Returns (out fp32, lse fp32).

    The aten op (unlike F.scaled_dot_product_attention) exposes the log-sum-exp
    needed to merge blocks exactly. CUDA fp16/bf16 only.
    """
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, scale=scale)[:2]
    return out.float(), lse


def _merge_block(out_acc, lse_acc, o_blk, lse_blk):
    """Online-softmax combine of two normalized partial attentions (all fp32).

    Initial state (out_acc=0, lse_acc=-inf) merges cleanly: exp(-inf - x) == 0.
    """
    lse_new = torch.logaddexp(lse_acc, lse_blk)
    out_acc = (
        out_acc * torch.exp(lse_acc - lse_new).unsqueeze(-1)
        + o_blk * torch.exp(lse_blk - lse_new).unsqueeze(-1)
    )
    return out_acc, lse_new


def _post_ring_rotate(cur_k, cur_v, recv_len, p2p_group, rank, world):
    """Post one non-blocking ring rotation: send the current K/V block to
    rank+1, receive the next block (length `recv_len`) from rank-1.

    K and V travel in a single batch_isend_irecv (the unbatched form is
    serialized by NCCL and contributed to the issue #3 deadlock), on the
    DEDICATED p2p group. Zero-length sends/receives are skipped on both
    endpoints — each consults the same all-gathered lengths, so the posted op
    sequence stays rank-symmetric. Exact lengths travel (no padding/mask).
    Returns (reqs, next_k, next_v); wait on reqs before reading next_k/next_v.
    """
    send_to = (rank + 1) % world
    recv_from = (rank - 1) % world
    B, H, _, Dh = cur_k.shape
    next_k = cur_k.new_empty((B, H, recv_len, Dh))
    next_v = cur_v.new_empty((B, H, recv_len, Dh))
    ops = []
    if cur_k.shape[2] > 0:
        ops += [
            dist.P2POp(dist.isend, cur_k, send_to, group=p2p_group),
            dist.P2POp(dist.isend, cur_v, send_to, group=p2p_group),
        ]
    if recv_len > 0:
        ops += [
            dist.P2POp(dist.irecv, next_k, recv_from, group=p2p_group),
            dist.P2POp(dist.irecv, next_v, recv_from, group=p2p_group),
        ]
    reqs = dist.batch_isend_irecv(ops) if ops else []
    return reqs, next_k, next_v


class RingAttention:
    """Blockwise flash/online-softmax attention; rotate K,V around the ring.

    Exact (== single-rank SDPA over the global sequence, up to fp reduction
    order) with O(local block) memory — no full score matrix and no full
    global K/V. CUDA half precision runs each ring step through the
    FlashAttention kernel; CPU/fp32 uses the chunked manual path. K and V
    rotate together in one batched isend/irecv per step on a DEDICATED
    process group (P2P sharing a communicator with the model's all_gathers
    deadlocked on 8x PCIe — issue #3), and each step's rotation is posted
    BEFORE that step's compute so communication overlaps it.
    """

    def __call__(self, q, k, v, cp_group) -> torch.Tensor:
        rank = dist.get_rank(cp_group)
        world = dist.get_world_size(cp_group)
        lengths = all_gather_ints(k.shape[2], cp_group, device=k.device)

        # Mirror F.sdpa's autocast handling: q/k/v arrive fp32 (q_norm/k_norm
        # emit fp32 under autocast); cast so the flash path can engage.
        if q.is_cuda and torch.is_autocast_enabled("cuda"):
            dt = torch.get_autocast_dtype("cuda")
            q, k, v = q.to(dt), k.to(dt), v.to(dt)

        B, H, Lq, Dh = q.shape
        scale = Dh ** -0.5
        use_flash = (
            q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)
            and Dh % 8 == 0 and Dh <= 256
        )
        block_attn = _block_attn_flash if use_flash else _block_attn_manual

        p2p = p2p_group_for(cp_group) if world > 1 else None
        out = torch.zeros((B, H, Lq, Dh), device=q.device, dtype=torch.float32)
        lse = torch.full((B, H, Lq), float("-inf"), device=q.device, dtype=torch.float32)

        cur_k, cur_v = k.contiguous(), v.contiguous()
        for step in range(world):
            if step < world - 1:  # post the next rotation; overlaps compute below
                recv_len = lengths[(rank - step - 1) % world]
                reqs, next_k, next_v = _post_ring_rotate(cur_k, cur_v, recv_len, p2p, rank, world)
            if lengths[(rank - step) % world] > 0 and Lq > 0:
                o_blk, lse_blk = block_attn(q, cur_k, cur_v, scale)
                out, lse = _merge_block(out, lse, o_blk, lse_blk)
            if step < world - 1:
                for r in reqs:
                    r.wait()
                cur_k, cur_v = next_k, next_v
        return out.to(q.dtype)


def build_strategy(name: str) -> DistributedAttention:
    strategies = {"all_gather_kv": AllGatherKVAttention, "ring": RingAttention}
    if name not in strategies:
        raise ValueError(f"Unknown cp_strategy {name!r}; choices: {sorted(strategies)}")
    return strategies[name]()
