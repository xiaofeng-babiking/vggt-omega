"""Distributed reductions for sharded evaluation.

Depth metrics: each rank scores its own frames; we all-reduce frame-weighted
sums so every rank ends with the single-GPU mean. Poses: gathered to rank 0,
ordered by global frame index, for the trajectory metric.
"""
import torch
import torch.distributed as dist


def reduce_depth_means(per_frame_metrics: list[dict], keys: list[str], group) -> dict:
    """Frame-count-weighted mean of each key across all ranks."""
    local_n = len(per_frame_metrics)
    sums = {k: float(sum(d[k] for d in per_frame_metrics)) for k in keys}
    device = "cuda" if dist.get_backend(group) == "nccl" else "cpu"
    packed = torch.tensor([float(local_n)] + [sums[k] for k in keys], device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)
    total_n = packed[0].item()
    return {k: (packed[i + 1].item() / total_n if total_n > 0 else 0.0) for i, k in enumerate(keys)}


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
    max_n = max(counts) if counts else 0

    dim = pose_enc_local.shape[-1]
    padded = pose_enc_local.new_zeros(1, max_n, dim)
    padded[:, :n_local] = pose_enc_local
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded, group=group)
    if rank != 0:
        return None
    parts = [g[:, :counts[r]] for r, g in enumerate(gathered)]
    return torch.cat(parts, dim=1)
