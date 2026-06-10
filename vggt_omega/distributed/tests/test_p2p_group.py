import torch
import torch.distributed as dist

from vggt_omega.distributed.process_group import p2p_group_for
from vggt_omega.distributed.tests._dist_test_util import run_distributed


def _worker(rank, world_size):
    g1 = p2p_group_for(dist.group.WORLD)
    g2 = p2p_group_for(dist.group.WORLD)
    assert g1 is g2, "p2p group must be cached per parent group"
    assert g1 is not dist.group.WORLD, "p2p group must be a separate communicator"
    # The dedicated group must carry batched P2P traffic.
    send = torch.full((2,), float(rank))
    recv = torch.empty(2)
    ops = [
        dist.P2POp(dist.isend, send, (rank + 1) % world_size, group=g1),
        dist.P2POp(dist.irecv, recv, (rank - 1) % world_size, group=g1),
    ]
    for r in dist.batch_isend_irecv(ops):
        r.wait()
    return recv


def test_p2p_group_dedicated_cached_and_functional():
    out = run_distributed(_worker, 3)
    for rank, recv in enumerate(out):
        assert recv[0].item() == float((rank - 1) % 3)
