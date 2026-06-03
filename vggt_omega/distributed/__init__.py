"""Multi-GPU sequence-parallel (Ulysses) inference for VGGT-Omega."""
from vggt_omega.distributed.parallel_context import (
    ParallelContext,
    init_sequence_parallel,
    install_sequence_parallel,
    shard_frames,
)

__all__ = [
    "ParallelContext",
    "init_sequence_parallel",
    "install_sequence_parallel",
    "shard_frames",
]
