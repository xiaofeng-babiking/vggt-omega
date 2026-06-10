"""Frame sharding and pad/mask helpers (pure; no torch.distributed calls)."""
import numpy as np
import torch


def frame_counts_for(num_frames: int, world_size: int) -> list[int]:
    """Per-rank frame counts: contiguous split, remainder to the lowest ranks."""
    base, rem = divmod(num_frames, world_size)
    return [base + (1 if r < rem else 0) for r in range(world_size)]


def shard_frame_ids(frame_ids: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """Contiguous frame-id slice owned by `rank` (possibly empty)."""
    counts = frame_counts_for(len(frame_ids), world_size)
    start = sum(counts[:rank])
    return frame_ids[start : start + counts[rank]]


def pad_seq_to(x: torch.Tensor, target_len: int, dim: int) -> torch.Tensor:
    """Zero-pad tensor `x` along `dim` up to `target_len` (no-op if already long enough)."""
    pad_amt = target_len - x.shape[dim]
    if pad_amt <= 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_amt
    return torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
