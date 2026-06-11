import numpy as np
import pytest

from vggt_omega.datasets.dynamic_dataloader import DynamicBatchSampler, DynamicDistributedSampler


def _make_sampler(rank, global_np_seed, epoch=0, seed=42, n_items=10000):
    """Sampler pair-builder simulating one DDP rank with a rank-dependent
    global numpy RNG (what Trainer._seed_everything does: seed + rank)."""
    np.random.seed(global_np_seed)
    dist_sampler = DynamicDistributedSampler(
        list(range(n_items)), num_replicas=2, rank=rank, shuffle=False
    )
    return DynamicBatchSampler(
        dist_sampler,
        aspect_ratio_range=[0.33, 1.0],
        image_num_range=[1, 24],
        epoch=epoch,
        seed=seed,
        max_img_per_gpu=24,
    )


def _draw_params(sampler, n_batches):
    out = []
    for i, batch in enumerate(sampler):
        if i >= n_batches:
            break
        idx, image_num, aspect = batch[0]
        out.append((image_num, aspect))
    return out


def test_frame_draws_identical_across_ranks():
    # Different global np seeds (rank-dependent) must NOT affect the
    # (image_num, aspect) sequence — only data indices may differ.
    a = _draw_params(_make_sampler(rank=0, global_np_seed=42), 200)
    b = _draw_params(_make_sampler(rank=1, global_np_seed=4242), 200)
    assert a == b


def test_frame_draws_deterministic_per_epoch_and_change_across_epochs():
    a = _draw_params(_make_sampler(rank=0, global_np_seed=1, epoch=0), 50)
    b = _draw_params(_make_sampler(rank=0, global_np_seed=2, epoch=0), 50)
    c = _draw_params(_make_sampler(rank=0, global_np_seed=1, epoch=1), 50)
    assert a == b          # same (seed, epoch) -> same sequence
    assert a != c          # epoch advances the stream


def test_image_num_marginal_stays_uniform():
    # mean batch size ~3.6 at cap 24, so 4800 batches consume ~17k indices:
    # give the per-rank shard enough items (n_items/2 per rank) to not run dry.
    draws = [s for s, _ in _draw_params(_make_sampler(rank=0, global_np_seed=7, n_items=200000), 4800)]
    counts = np.bincount(draws, minlength=25)[1:25]
    assert counts.min() > 0
    # uniform expectation 200/bin; allow generous sampling noise
    assert counts.max() / counts.min() < 1.6
