import torch.distributed as dist

from vggt_omega.distributed.eval_reduce import (
    gather_pose_enc_to_rank0,
    reduce_depth,
)
from vggt_omega.evaluates import scene
from vggt_omega.distributed.tests._dist_test_util import run_distributed


# Per-rank per-frame depth dicts with UNEVEN shards (incl. an empty rank), so the
# test proves frame-weighting (not mean-of-rank-means) and empty-shard safety.
def _shard_for(rank):
    allframes = [
        [{k: 1.0 for k in scene.DEPTH_KEYS}, {k: 2.0 for k in scene.DEPTH_KEYS}],  # rank 0: 2 frames
        [{k: 3.0 for k in scene.DEPTH_KEYS}],                                       # rank 1: 1 frame
        [],                                                                          # rank 2: empty
        [{k: 4.0 for k in scene.DEPTH_KEYS}, {k: 5.0 for k in scene.DEPTH_KEYS},
         {k: 6.0 for k in scene.DEPTH_KEYS}],                                       # rank 3: 3 frames
    ]
    return allframes[rank]


def _worker(rank, world_size):
    out = reduce_depth(_shard_for(rank), dist.group.WORLD)
    # global frame-weighted mean of 1..6 = 3.5, over 6 scored frames; same on every rank.
    for k in scene.DEPTH_KEYS:
        assert abs(out[k] - 3.5) < 1e-9, (k, out[k])
    assert out["num_frames"] == 6
    return out["num_frames"]


def test_reduce_depth_frame_weighted_over_uneven_shards():
    results = run_distributed(_worker, 4)
    assert all(r == 6 for r in results)


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


def _pose_worker_empty(rank, world_size):
    import torch
    counts = [2, 0, 1]  # rank 1 empty
    start = sum(counts[:rank])
    n = counts[rank]
    pose = torch.arange(start, start + n, dtype=torch.float32).reshape(1, n, 1).expand(1, n, 9).contiguous()
    return gather_pose_enc_to_rank0(pose, dist.group.WORLD)


def test_gather_pose_enc_with_empty_rank():
    results = run_distributed(_pose_worker_empty, 3)
    assert results[0].shape == (1, 3, 9)
    assert results[0][0, :, 0].tolist() == [0, 1, 2]  # rank1's empty shard contributes nothing
