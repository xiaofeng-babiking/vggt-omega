"""Sim(3) fit + pose helpers: the properties the off-the-grid gate relies on."""
import torch

from vggt_omega.utils.geometry import (
    apply_sim3_to_extrinsics,
    camera_centers,
    umeyama_sim3,
)


def _random_rotation(generator=None):
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=generator))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_umeyama_recovers_known_sim3():
    torch.manual_seed(0)
    src = torch.randn(20, 3, dtype=torch.float64)
    rotation_gt = _random_rotation()
    scale_gt = 1.7
    translation_gt = torch.tensor([0.3, -1.2, 2.0], dtype=torch.float64)
    dst = scale_gt * src @ rotation_gt.T + translation_gt

    scale, rotation, translation = umeyama_sim3(src, dst)
    assert torch.allclose(scale, torch.tensor(scale_gt, dtype=torch.float64), atol=1e-9)
    assert torch.allclose(rotation, rotation_gt, atol=1e-9)
    assert torch.allclose(translation, translation_gt, atol=1e-9)
    fit = scale * src @ rotation.T + translation
    assert torch.allclose(fit, dst, atol=1e-9)


def test_umeyama_preserves_dtype_and_device():
    src = torch.randn(10, 3, dtype=torch.float32)
    dst = src * 2.0 + 1.0
    scale, rotation, translation = umeyama_sim3(src, dst)
    assert scale.dtype == torch.float32
    assert rotation.shape == (3, 3)
    assert torch.allclose(scale, torch.tensor(2.0), atol=1e-5)


def test_umeyama_degenerate_inputs_fall_back_to_identity_rotation():
    eye = torch.eye(3, dtype=torch.float64)
    # Empty input: identity Sim(3).
    scale, rotation, translation = umeyama_sim3(
        torch.zeros(0, 3, dtype=torch.float64), torch.zeros(0, 3, dtype=torch.float64)
    )
    assert float(scale) == 1.0 and torch.equal(rotation, eye)

    # Coincident points: no rotation is determined; median-baseline scale.
    src = torch.zeros(5, 3, dtype=torch.float64)
    dst = torch.ones(5, 3, dtype=torch.float64)
    scale, rotation, translation = umeyama_sim3(src, dst)
    assert torch.equal(rotation, eye)

    # Colinear points (rank 1): same fallback, scale from baselines.
    line = torch.linspace(-1, 1, 7, dtype=torch.float64)[:, None] * torch.tensor(
        [[1.0, 0.0, 0.0]], dtype=torch.float64
    )
    scale, rotation, translation = umeyama_sim3(line, 3.0 * line)
    assert torch.equal(rotation, eye)
    assert torch.allclose(scale, torch.tensor(3.0, dtype=torch.float64))


def test_umeyama_nonfinite_input_survives_and_falls_back():
    src = torch.randn(8, 3, dtype=torch.float64)
    dst = src.clone()
    dst[3, 1] = float("nan")
    scale, rotation, translation = umeyama_sim3(src, dst)  # must not raise
    assert torch.equal(rotation, torch.eye(3, dtype=torch.float64))


def test_apply_sim3_moves_camera_centers_by_the_sim3():
    torch.manual_seed(1)
    rot_cam = _random_rotation().float()
    center = torch.tensor([0.5, -0.2, 1.0])
    extr = torch.cat([rot_cam, (-rot_cam @ center)[:, None]], dim=-1)[None]

    rotation = _random_rotation().float()
    scale = torch.tensor(1.3)
    translation = torch.tensor([0.1, 0.2, -0.3])

    moved = apply_sim3_to_extrinsics(extr, scale, rotation, translation)
    expected_center = scale * rotation @ center + translation
    assert torch.allclose(camera_centers(moved)[0], expected_center, atol=1e-5)
    # Rotation block stays orthonormal (the 1/s factor is deliberately dropped).
    r = moved[0, :3, :3]
    assert torch.allclose(r @ r.T, torch.eye(3), atol=1e-5)
