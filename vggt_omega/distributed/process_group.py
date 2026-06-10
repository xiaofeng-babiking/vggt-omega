"""Context-parallel process-group setup and small collective helpers.

For inference the CP group spans the whole world (one sequence at a time, no DP).
"""
import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """Init NCCL from torchrun env vars. Returns (rank, world_size, local_rank)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Bind the group to this rank's device: pins NCCL to the right GPU and
        # mutes the "barrier(): using the device under current context" warning.
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", local_rank)
        )
    return rank, world_size, local_rank


def cp_group() -> "dist.ProcessGroup | None":
    """The context-parallel group (== world group for inference)."""
    return dist.group.WORLD


def all_gather_ints(value: int, group, device) -> list[int]:
    """All-gather one int per rank into an ordered Python list."""
    world = dist.get_world_size(group)
    t = torch.tensor([value], dtype=torch.long, device=device)
    out = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(out, t, group=group)
    return [int(x.item()) for x in out]


_p2p_groups: dict = {}


def p2p_group_for(group) -> "dist.ProcessGroup":
    """Dedicated process group (own communicator) for ring P2P traffic.

    NCCL may schedule send/recv and collectives that share a communicator
    differently across ranks, which deadlocked the full multi-block model on
    8x PCIe (issue #3); an isolated communicator removes that interaction.
    Creation is collective: the first call must happen at a rank-symmetric
    point (the ring strategy guarantees this). Cached per parent group.
    """
    pg = _p2p_groups.get(group)
    if pg is None:
        pg = dist.new_group(ranks=dist.get_process_group_ranks(group))
        dist.barrier(group=pg)  # force eager communicator init, symmetrically
        _p2p_groups[group] = pg
    return pg
