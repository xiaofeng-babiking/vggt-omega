import torch
import torch.distributed as dist

from vggt_omega.distributed.attention import AllGatherKVAttention
from vggt_omega.distributed.camera_head import ContextParallelCameraHead
from vggt_omega.distributed.tests._dist_test_util import init_finite, run_distributed
from vggt_omega.models.heads import CameraHead

DIM_IN, NUM_FRAMES, PTS = 32, 6, 3  # patch_token_start = 1 + num_register


def _make_head():
    torch.manual_seed(13)
    head = CameraHead(dim_in=DIM_IN).eval()
    # Initialize all params (LayerScale.gamma etc. are torch.empty until reset) and
    # set the NaN-by-design mask_k_bias to ones, so the from-scratch head is finite.
    return init_finite(head)


def _make_tokens(num_frames, num_tokens):
    g = torch.Generator().manual_seed(8)
    # last cached layer only is read by CameraHead; build a minimal list of length 1.
    return [torch.randn(1, num_frames, num_tokens, DIM_IN, generator=g)]


def _worker(rank, world_size, state_dict, counts, num_tokens):
    head = ContextParallelCameraHead(dim_in=DIM_IN).eval()
    head.load_state_dict(state_dict)
    head.cp_group = dist.group.WORLD
    head.strategy = AllGatherKVAttention()
    tokens = _make_tokens(NUM_FRAMES, num_tokens)
    start = sum(counts[:rank])
    local = [tokens[-1][:, start : start + counts[rank]].contiguous()]
    with torch.no_grad():
        return head(local, patch_token_start=PTS)


def test_cp_camera_head_matches_base():
    num_tokens = PTS + 4  # a few patch tokens beyond cam+reg
    head = _make_head()
    tokens = _make_tokens(NUM_FRAMES, num_tokens)
    with torch.no_grad():
        ref = head(tokens, patch_token_start=PTS)  # (1, N, 9)
    parts = run_distributed(_worker, 3, head.state_dict(), [2, 2, 2], num_tokens)
    got = torch.cat(parts, dim=1)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)
