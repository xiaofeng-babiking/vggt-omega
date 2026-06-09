import torch.distributed as dist

from vggt_omega.distributed.process_group import all_gather_ints
from vggt_omega.distributed.tests._dist_test_util import run_distributed


def _gather_worker(rank, world_size):
    # each rank contributes (rank + 1); expect [1, 2, ...]
    return all_gather_ints(rank + 1, dist.group.WORLD, device="cpu")


def test_all_gather_ints_collects_per_rank_values():
    results = run_distributed(_gather_worker, 3)
    for per_rank in results:
        assert per_rank == [1, 2, 3]
