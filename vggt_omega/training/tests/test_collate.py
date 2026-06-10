import pytest
import torch

from vggt_omega.training.collate import train_collate


def _fake_sample(seq="seq_a", extra=None, tracks=False):
    num_frames, height, width = 2, 8, 8
    sample = {
        "seq_name": seq,
        "ids": torch.arange(num_frames),
        "images": torch.rand(num_frames, 3, height, width),
        "depths": torch.rand(num_frames, height, width),
        "extrinsics": torch.rand(num_frames, 3, 4),
        "intrinsics": torch.rand(num_frames, 3, 3),
        "cam_points": torch.rand(num_frames, 3, height * width),
        "world_points": torch.rand(num_frames, height, width, 3),
        "point_masks": torch.ones(num_frames, height, width, dtype=torch.bool),
        "modalities": ["DEPTH", "POSE"],
    }
    if tracks:
        track_num = 4
        sample["tracks"] = torch.rand(num_frames, track_num, 2)
        sample["track_vis_mask"] = torch.ones(num_frames, track_num, dtype=torch.bool)
        sample["track_positive_mask"] = torch.ones(track_num, dtype=torch.bool)
    sample.update(extra or {})
    return sample


def test_collate_stacks_core_and_drops_unshared_extras():
    a = _fake_sample(seq="tum_x", extra={"sky_masks": torch.zeros(2, 8, 8, dtype=torch.bool)})
    b = _fake_sample(seq="co3d_y", extra={})            # no sky_masks
    out = train_collate([a, b])
    assert out["images"].shape[0] == 2
    assert "sky_masks" not in out                       # unshared -> dropped
    assert out["seq_name"] == ["tum_x", "co3d_y"]
    assert isinstance(out["modalities"], list) and len(out["modalities"]) == 2


def test_collate_keeps_shared_tracks():
    a, b = _fake_sample(tracks=True), _fake_sample(tracks=True)
    out = train_collate([a, b])
    assert out["tracks"].shape[0] == 2 and out["track_positive_mask"].dtype == torch.bool


def test_collate_requires_core_keys():
    with pytest.raises(KeyError):
        train_collate([{"images": torch.zeros(2, 3, 8, 8)}])
