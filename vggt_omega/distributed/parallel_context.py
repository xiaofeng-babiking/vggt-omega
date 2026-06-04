"""Sequence-parallel context: the device mesh, the ``sp`` process group, and
the collective wrappers the model calls. Built around a named-axis
``DeviceMesh`` so a ``"tp"`` axis can be added later without touching this code.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from vggt_omega.distributed import ulysses


class ParallelContext:
    """Holds the ``sp`` group and the collectives used by SP-aware modules."""

    def __init__(self, mesh: DeviceMesh) -> None:
        self.mesh = mesh
        self.sp_group = mesh.get_group("sp")
        self.sp_size = dist.get_world_size(self.sp_group)
        self.sp_rank = dist.get_rank(self.sp_group)

    @property
    def is_first_shard(self) -> bool:
        """True for the rank owning the global first frame (contiguous sharding)."""
        return self.sp_rank == 0

    # -- attention: token-shard <-> head-shard ------------------------------
    def scatter_heads(self, x: torch.Tensor) -> torch.Tensor:
        return ulysses.scatter_heads(x, self.sp_group, self.sp_size)

    def gather_heads(self, x: torch.Tensor) -> torch.Tensor:
        return ulysses.gather_heads(x, self.sp_group, self.sp_size)

    # -- special tokens: gather all frames, slice this shard back -----------
    def all_gather_frames(self, x: torch.Tensor, frame_dim: int = 1) -> torch.Tensor:
        """``(.., S_local, ..)`` -> ``(.., S_full, ..)`` along ``frame_dim``."""
        chunks = [torch.empty_like(x) for _ in range(self.sp_size)]
        dist.all_gather(chunks, x.contiguous(), group=self.sp_group)
        return torch.cat(chunks, dim=frame_dim)

    def slice_local_frames(self, x_full: torch.Tensor, frame_dim: int = 1) -> torch.Tensor:
        s_local = x_full.shape[frame_dim] // self.sp_size
        return x_full.narrow(frame_dim, self.sp_rank * s_local, s_local)


def install_sequence_parallel(model: torch.nn.Module, ctx: ParallelContext | None) -> None:
    """Wire ``ctx`` into the SP-aware modules. ``ctx=None`` -> single-GPU (no-op)."""
    if ctx is not None and getattr(model, "text_alignment_head", None) is not None:
        # TextAlignmentHead runs a cross-frame readout over the special tokens
        # (like the camera trunk) but is NOT sequence-parallel-aware: under
        # frame-sharding it would silently see only the local shard's frames.
        # Fail loudly rather than return wrong embeddings.
        raise NotImplementedError(
            "TextAlignmentHead is not sequence-parallel-aware; disable it "
            "(enable_alignment=False) for multi-GPU inference, or make its "
            "cross-frame readout gather special tokens first."
        )
    agg = model.aggregator
    agg.seq_parallel_ctx = ctx
    for block, atype in zip(agg.inter_frame_blocks, agg.inter_frame_attention_types):
        # Only the GLOBAL blocks all-to-all inside attention. Register blocks
        # gather their special tokens at the aggregator level, so their
        # attention stays local (ctx left None).
        block.attn.seq_parallel_ctx = ctx if atype == "global" else None
    if getattr(model, "camera_head", None) is not None:
        model.camera_head.seq_parallel_ctx = ctx


def init_sequence_parallel() -> ParallelContext | None:
    """Init from a ``torchrun`` env. Returns None for single-process runs."""
    if "RANK" not in os.environ:
        return None
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if world_size == 1:
        return None
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("sp",))
    return ParallelContext(mesh)


def shard_frames(frame_ids, sp_size: int, sp_rank: int):
    """Contiguous frame block for ``sp_rank``. Requires len divisible by sp_size.

    Works on a list or a numpy array (returns the same type via slicing).
    """
    n = len(frame_ids)
    if n % sp_size != 0:
        raise ValueError(
            f"num_frames ({n}) must be divisible by sequence-parallel world_size "
            f"({sp_size}); pick a multiple of {sp_size}."
        )
    per = n // sp_size
    return frame_ids[sp_rank * per : (sp_rank + 1) * per]
