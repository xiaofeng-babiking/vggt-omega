import torch.distributed as dist

from vggt_omega.distributed.eval_reduce import (
    gather_pose_enc_to_rank0,
    reduce_depth_means,
)
from vggt_omega.distributed.tests._dist_test_util import run_distributed


def _depth_worker(rank, world_size):
    # rank r contributes r+1 frames each with abs_rel = (r+1)*10 (distinct, easy to verify)
    n = rank + 1
    per_frame = [{"abs_rel": float((rank + 1) * 10)} for _ in range(n)]
    return reduce_depth_means(per_frame, ["abs_rel"], dist.group.WORLD)


def test_reduce_depth_means_is_frame_weighted():
    # frames: rank0=[10], rank1=[20,20] -> mean = (10 + 20 + 20)/3
    results = run_distributed(_depth_worker, 2)
    for r in results:
        assert abs(r["abs_rel"] - (10 + 20 + 20) / 3) < 1e-6


def _pose_worker(rank, world_size, counts):
    import torch
    start = sum(counts[:rank])
    n = counts[rank]
    pose = torch.arange(start, start + n, dtype=torch.float32).reshape(1, n, 1).expand(1, n, 9).contiguous()
    return gather_pose_enc_to_rank0(pose, dist.group.WORLD)


def test_gather_pose_enc_orders_by_global_index():
    results = run_distributed(_pose_worker, 3, [2, 2, 2])
    rank0 = results[0]
    assert rank0.shape == (1, 6, 9)
    assert rank0[0, :, 0].tolist() == [0, 1, 2, 3, 4, 5]
    for r in (1, 2):
        assert results[r] is None
