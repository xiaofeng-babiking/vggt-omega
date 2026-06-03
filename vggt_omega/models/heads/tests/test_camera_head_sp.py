"""CameraHead (frame-sharded) matches single-GPU pose_enc. CameraHead has no
heavy patch-embed backbone, so it is constructed directly at small width."""
from __future__ import annotations

import torch
import pytest

from vggt_omega.models.heads.camera_head import CameraHead
from vggt_omega.models.layers.vision_transformer import init_weights_vit
from vggt_omega.distributed.tests._dist import run_distributed

skip_no_multigpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs >=2 CUDA devices + NCCL"
)


def _camera_head_parity(rank: int, world_size: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from vggt_omega.distributed import ParallelContext

    dev = f"cuda:{rank}"
    dim_in, S, T = 64, 4, 20  # 2*embed_dim=64; S frames; T>=patch_token_start tokens
    torch.manual_seed(0)  # identical weights on every rank
    head = CameraHead(dim_in=dim_in).to(dev).eval()
    head.apply(init_weights_vit)  # set masked-bias buffers (mask_k_bias=True -> bias_mask=NaN otherwise)
    # init_weights_vit sets the trunk's LayerScale gamma to 1e-5, which suppresses
    # the cross-frame attention to ~0 and makes the gather undetectable. Boost it to
    # O(1) so cross-frame mixing actually affects the output -> the test can tell a
    # correct gather from a missing one (see the negative control below).
    for block in head.trunk:
        torch.nn.init.constant_(block.ls1.gamma, 1.0)
        torch.nn.init.constant_(block.ls2.gamma, 1.0)
    # init_weights_vit also initialises camera_branch linears with std=0.02 (~0),
    # which collapses the ~0.013 cross-frame delta at the trunk token level down to
    # ~1e-5 at the final 9-dim output — below atol=1e-3. Boost to std=1.0 so the
    # camera_branch faithfully propagates the cross-frame signal to the output.
    for m in head.camera_branch.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    torch.manual_seed(1)  # identical full input on every rank
    tokens_full = torch.randn(1, S, T, dim_in, device=dev)
    with torch.inference_mode():
        ref = head([tokens_full], patch_token_start=17)  # full cross-frame context, (1, S, 9)

    s_local = S // world_size
    sl = slice(rank * s_local, (rank + 1) * s_local)
    local = tokens_full[:, sl].contiguous()

    # Negative control: WITHOUT the gather (ctx=None), each shard's trunk sees only
    # its local frames, so the cross-frame attention differs -> the output must NOT
    # match the full-context reference. This proves the test has teeth.
    head.seq_parallel_ctx = None
    with torch.inference_mode():
        no_gather = head([local], patch_token_start=17)
    assert not torch.allclose(
        no_gather, ref[:, sl], atol=1e-3
    ), "vacuous test: cross-frame attention has no measurable effect"

    # Positive: WITH the gather, the sharded forward matches the full-context ref.
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("sp",))
    head.seq_parallel_ctx = ParallelContext(mesh)
    with torch.inference_mode():
        out_local = head([local], patch_token_start=17)  # (1, S_local, 9)
    assert torch.allclose(out_local, ref[:, sl], atol=1e-3)


@skip_no_multigpu
def test_camera_head_parity():
    run_distributed(_camera_head_parity, world_size=2)
