"""ParallelContext collectives: frame gather/slice round-trip + shard_frames."""
from __future__ import annotations

import torch
import torch.distributed as dist
import pytest

from vggt_omega.distributed.tests._dist import run_distributed

skip_no_multigpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs >=2 CUDA devices + NCCL"
)


def _frames_roundtrip(rank: int, world_size: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from vggt_omega.distributed import ParallelContext

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("sp",))
    ctx = ParallelContext(mesh)
    # Each rank owns a distinct frame block: value == global frame index.
    s_local = 2
    local = torch.arange(
        rank * s_local, (rank + 1) * s_local, device=f"cuda:{rank}", dtype=torch.float32
    ).reshape(1, s_local, 1, 1)
    full = ctx.all_gather_frames(local, frame_dim=1)
    assert full.shape[1] == world_size * s_local
    expected = torch.arange(
        world_size * s_local, device=f"cuda:{rank}", dtype=torch.float32
    ).reshape(1, -1, 1, 1)
    assert torch.equal(full, expected)
    assert torch.equal(ctx.slice_local_frames(full, frame_dim=1), local)
    assert ctx.is_first_shard == (rank == 0)


@skip_no_multigpu
def test_frames_gather_slice_roundtrip():
    run_distributed(_frames_roundtrip, world_size=2)


def test_shard_frames_divides_contiguously():
    from vggt_omega.distributed import shard_frames

    ids = list(range(8))
    assert shard_frames(ids, sp_size=4, sp_rank=0) == [0, 1]
    assert shard_frames(ids, sp_size=4, sp_rank=3) == [6, 7]


def test_shard_frames_requires_divisible():
    from vggt_omega.distributed import shard_frames

    with pytest.raises(ValueError):
        shard_frames(list(range(7)), sp_size=2, sp_rank=0)


def test_install_sets_ctx_on_global_blocks_only():
    import types
    from vggt_omega.distributed import install_sequence_parallel

    def fake_attn():
        return types.SimpleNamespace(seq_parallel_ctx="UNSET")

    blocks = [types.SimpleNamespace(attn=fake_attn()) for _ in range(6)]
    atypes = ["global", "global", "register", "global", "register", "global"]
    aggregator = types.SimpleNamespace(
        seq_parallel_ctx=None, inter_frame_blocks=blocks, inter_frame_attention_types=atypes
    )
    camera_head = types.SimpleNamespace(seq_parallel_ctx="UNSET")
    model = types.SimpleNamespace(aggregator=aggregator, camera_head=camera_head)

    sentinel = object()
    install_sequence_parallel(model, sentinel)

    assert aggregator.seq_parallel_ctx is sentinel
    assert camera_head.seq_parallel_ctx is sentinel
    for block, atype in zip(blocks, atypes):
        assert block.attn.seq_parallel_ctx is (sentinel if atype == "global" else None)


def test_install_rejects_enabled_text_alignment_head():
    import types
    from vggt_omega.distributed import install_sequence_parallel

    aggregator = types.SimpleNamespace(
        seq_parallel_ctx=None, inter_frame_blocks=[], inter_frame_attention_types=[]
    )
    model = types.SimpleNamespace(
        aggregator=aggregator, camera_head=None, text_alignment_head=object()
    )
    # The alignment head is not SP-aware -> installing a context must fail loudly.
    with pytest.raises(NotImplementedError):
        install_sequence_parallel(model, object())
    # ctx=None (single-GPU) is always allowed, even with alignment enabled.
    install_sequence_parallel(model, None)
