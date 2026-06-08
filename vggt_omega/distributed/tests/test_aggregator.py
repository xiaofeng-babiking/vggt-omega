import torch
import torch.distributed as dist

from vggt_omega.distributed.aggregator import ContextParallelAggregator
from vggt_omega.distributed.attention import AllGatherKVAttention
from vggt_omega.distributed.tests._dist_test_util import run_distributed
from vggt_omega.models.aggregator import Aggregator

# Tiny aggregator config for speed (real defaults: depth=24, embed_dim=1024).
CFG = dict(patch_size=16, embed_dim=64, depth=4, num_heads=4,
           num_register_tokens=2, register_attention_block_indices=[1],
           cached_layer_indices=(1, 3))
NUM_FRAMES, IMG = 6, 32  # 32/16 = 2x2 patch grid


def _make_aggregator():
    torch.manual_seed(11)
    agg = Aggregator(**CFG).eval()
    for m in agg.modules():
        if hasattr(m, "bias_mask"):
            torch.nn.init.ones_(m.bias_mask)
    return agg


def _make_images(num_frames):
    g = torch.Generator().manual_seed(5)
    return torch.rand(1, num_frames, 3, IMG, IMG, generator=g)


def _worker(rank, world_size, state_dict, counts):
    agg = ContextParallelAggregator(**CFG).eval()
    agg.load_state_dict(state_dict)
    agg.cp_group = dist.group.WORLD
    agg.strategy = AllGatherKVAttention()
    images = _make_images(NUM_FRAMES)
    start = sum(counts[:rank])
    local = images[:, start : start + counts[rank]].contiguous()
    with torch.no_grad():
        outputs, ps = agg(local)
    # Return the final cached layer (B, n_local, tokens, 2*embed) + patch_start.
    return outputs[-1], ps


def test_cp_aggregator_matches_base():
    base = _make_aggregator()
    images = _make_images(NUM_FRAMES)
    with torch.no_grad():
        ref_outputs, ref_ps = base(images)
    ref_final = ref_outputs[-1]

    parts = run_distributed(_worker, 3, base.state_dict(), [2, 2, 2])
    got_final = torch.cat([p[0] for p in parts], dim=1)
    assert all(p[1] == ref_ps for p in parts)
    torch.testing.assert_close(got_final, ref_final, atol=1e-4, rtol=1e-4)


def test_cp_aggregator_uneven():
    base = _make_aggregator()
    images = _make_images(NUM_FRAMES)
    with torch.no_grad():
        ref_final = base(images)[0][-1]
    parts = run_distributed(_worker, 4, base.state_dict(), [2, 2, 1, 1])
    got_final = torch.cat([p[0] for p in parts], dim=1)
    torch.testing.assert_close(got_final, ref_final, atol=1e-4, rtol=1e-4)
