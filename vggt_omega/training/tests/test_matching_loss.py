import torch

from vggt_omega.training.losses import TrainLossComputer, matching_loss
from vggt_omega.training.tests.conftest import (
    SCENE_H,
    SCENE_W,
    _intrinsics_for_scene,
    _random_consistent_scene,
)


def _separable_setup():
    C, S, G = 8, 2, 4
    tok = torch.zeros(1, S, G * G, C)
    eye = torch.eye(G * G, C)
    tok[0, 0] = eye
    tok[0, 1] = eye
    tracks = torch.tensor(
        [[[8.0, 8.0], [24.0, 8.0]], [[8.0, 8.0], [24.0, 8.0]]]
    ).unsqueeze(0)
    vis = torch.ones(1, S, 2, dtype=torch.bool)
    pos = torch.tensor([[True, True]])
    return tok, tracks, vis, pos


def test_matching_loss_separable_tokens():
    tok, tracks, vis, pos = _separable_setup()
    l_match = matching_loss(tok, tracks, vis, pos, patch_size=16, image_size_hw=(64, 64))
    assert torch.allclose(l_match, torch.tensor(0.31326), atol=1e-3)


def test_matching_loss_negative_pairs_pushed_apart():
    tok_identical, tracks, vis, _ = _separable_setup()
    pos = torch.tensor([[False, False]])
    tok_orthogonal = tok_identical.clone()
    tok_orthogonal[0, 1, 0] = torch.eye(16, 8)[2]
    tok_orthogonal[0, 1, 1] = torch.eye(16, 8)[3]
    l_identical = matching_loss(tok_identical, tracks, vis, pos, 16, (64, 64))
    l_orthogonal = matching_loss(tok_orthogonal, tracks, vis, pos, 16, (64, 64))
    assert l_identical > l_orthogonal


def test_matching_loss_skips_single_frame_and_empty():
    tok, tracks, vis, pos = _separable_setup()
    l = matching_loss(tok[:, :1], tracks[:, :1], vis[:, :1], pos, 16, (64, 64))
    assert l.item() == 0.0 and l.dtype == torch.float32
    no_pairs = matching_loss(
        tok, tracks, torch.zeros_like(vis), torch.ones(1, 2, dtype=torch.bool), 16, (64, 64)
    )
    assert no_pairs.item() == 0.0


def test_matching_loss_bounds_checks_negatives():
    tok, tracks, vis, _ = _separable_setup()
    tracks_oob = tracks.clone()
    tracks_oob[0, 1, 0] = torch.tensor([200.0, 8.0])
    l = matching_loss(tok, tracks_oob, vis, torch.tensor([[False, False]]), 16, (64, 64))
    assert torch.isfinite(l)


def test_matching_loss_casts_low_precision_tokens():
    tok, tracks, vis, pos = _separable_setup()
    l = matching_loss(tok.to(torch.bfloat16), tracks, vis, pos, 16, (64, 64))
    assert l.dtype == torch.float32 and torch.isfinite(l)


def test_train_loss_computer_end_to_end():
    from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding

    torch.manual_seed(0)
    B, S, H, W = 1, 3, SCENE_H, SCENE_W
    ext, dep, wp, mask = _random_consistent_scene(B=B, S=S)
    K = _intrinsics_for_scene(B=B, S=S)
    gt_enc = extri_intri_to_pose_encoding(ext, K, (H, W))
    batch = {
        "extrinsics": ext,
        "depths": dep,
        "intrinsics": K,
        "world_points": wp,
        "point_masks": mask,
        "tracks": torch.rand(B, S, 4, 2) * torch.tensor([W - 1.0, H - 1.0]),
        "track_vis_mask": torch.tensor([[[True, True, False, False]] * S]),
        "track_positive_mask": torch.tensor([[True, True, False, False]]),
    }
    P = (H // 16) * (W // 16)
    predictions = {
        "pose_enc": gt_enc + 0.05 * torch.randn_like(gt_enc),
        "depth": (torch.rand(B, S, H, W, 1) + 0.5),
        "depth_conf": 1.0 + torch.rand(B, S, H, W),
        "patch_tokens": torch.randn(B, S, P, 8),
    }
    computer = TrainLossComputer(
        weights={"camera": 5.0, "depth": 1.0, "point": 0.5, "match": 0.1}
    )
    out = computer(predictions, batch, image_size_hw=(H, W))
    for k in ("total", "camera", "depth", "point", "match"):
        assert k in out and torch.isfinite(out[k])
    assert "gt_scale" in out and torch.isfinite(out["gt_scale"])
    assert torch.allclose(
        out["total"],
        5.0 * out["camera"] + 1.0 * out["depth"] + 0.5 * out["point"] + 0.1 * out["match"],
        atol=1e-5,
    )


def test_train_loss_computer_zero_match_without_tracks():
    from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding

    B, S, H, W = 1, 2, SCENE_H, SCENE_W
    ext, dep, wp, mask = _random_consistent_scene(B=B, S=S)
    K = _intrinsics_for_scene(B=B, S=S)
    batch = {
        "extrinsics": ext,
        "depths": dep,
        "intrinsics": K,
        "world_points": wp,
        "point_masks": mask,
    }
    predictions = {
        "pose_enc": extri_intri_to_pose_encoding(ext, K, (H, W)),
        "depth": dep.unsqueeze(-1),
        "depth_conf": torch.ones(B, S, H, W),
    }
    computer = TrainLossComputer(
        weights={"camera": 5.0, "depth": 1.0, "point": 0.5, "match": 0.1}
    )
    out = computer(predictions, batch, image_size_hw=(H, W))
    assert out["match"].item() == 0.0
    assert torch.isfinite(out["total"])
