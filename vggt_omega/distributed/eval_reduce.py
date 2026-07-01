"""Distributed reductions for sharded evaluation.

Depth metrics: each rank scores its own frames; we all-reduce frame-weighted
sums so every rank ends with the single-GPU mean. Poses: gathered to rank 0,
ordered by global frame index, for the trajectory metric.
"""
import torch
import torch.distributed as dist


def reduce_depth(per_frame_local: list[dict], group) -> dict | None:
    """Frame-weighted mono-depth means across ranks: sum each depth key + the
    scored-frame count, all-reduce, then aggregate to the single-GPU schema via
    the shared scene aggregator. Every rank returns the same dict (or None if no
    rank scored a frame), so uneven shards yield exactly the single-GPU mean."""
    from vggt_omega.evaluates.scene import depth_sums, aggregate_depth_from_sums

    sums, count = depth_sums(per_frame_local)
    keys = list(sums.keys())
    device = "cuda" if dist.get_backend(group) == "nccl" else "cpu"
    packed = torch.tensor([float(count)] + [sums[k] for k in keys], device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
    total = int(round(packed[0].item()))
    reduced_sums = {k: packed[i + 1].item() for i, k in enumerate(keys)}
    return aggregate_depth_from_sums(reduced_sums, total)


def gather_pose_enc_to_rank0(pose_enc_local: torch.Tensor, group) -> torch.Tensor | None:
    """Gather per-rank (1, n_local, 9) pose encodings to rank 0 as (1, N, 9), in global order.

    Ranks are contiguous frame shards, so rank order == global frame order.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    n_local = pose_enc_local.shape[1]
    counts = [torch.zeros(1, dtype=torch.long, device=pose_enc_local.device) for _ in range(world)]
    dist.all_gather(counts, torch.tensor([n_local], dtype=torch.long, device=pose_enc_local.device), group=group)
    counts = [int(c.item()) for c in counts]
    max_n = max(counts)

    dim = pose_enc_local.shape[-1]
    padded = pose_enc_local.new_zeros(1, max_n, dim)
    padded[:, :n_local] = pose_enc_local
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded, group=group)
    if rank != 0:
        return None
    parts = [g[:, :counts[r]] for r, g in enumerate(gathered)]
    return torch.cat(parts, dim=1)
