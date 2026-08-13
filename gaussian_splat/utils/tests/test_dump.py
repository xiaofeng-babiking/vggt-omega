"""CPU round-trip tests for the SuperSplat-compatible .ply dump."""

import numpy as np
import pytest
import torch

from vggt_omega.models.heads import Gaussians
from vggt_omega.models.heads.gaussian_head import build_covariance

from vggt_omega.models.heads.gaussian_head import GSDecoder

from gaussian_splat.utils import dump_gaussians_ply, dump_per_frame_ply, dump_ply

_N = 37
_D_SH = 4  # sh_degree 1


def _random_inputs(seed: int = 0):
    torch.manual_seed(seed)
    means = torch.randn(_N, 3)
    scales = torch.rand(_N, 3) * 0.1 + 1e-3
    rotations = torch.nn.functional.normalize(torch.randn(_N, 4), dim=-1)
    harmonics = torch.randn(_N, 3, _D_SH)
    opacities = torch.rand(_N) * 0.98 + 0.01
    return means, scales, rotations, harmonics, opacities


def _read_ply(path):
    """Parse header + binary payload of a float32 little-endian INRIA ply."""
    blob = open(path, "rb").read()
    header, _, payload = blob.partition(b"end_header\n")
    lines = header.decode("ascii").splitlines()
    assert lines[0] == "ply"
    assert lines[1] == "format binary_little_endian 1.0"
    assert lines[2].startswith("element vertex ")
    count = int(lines[2].split()[-1])
    names = []
    for line in lines[3:]:
        kind, dtype, name = line.split()
        assert (kind, dtype) == ("property", "float")
        names.append(name)
    data = np.frombuffer(payload, dtype=np.dtype([(n, "<f4") for n in names]), count=count)
    return names, data


def _field(data, names):
    return np.stack([data[n] for n in names], axis=1)


def test_round_trip_recovers_activated_values(tmp_path):
    means, scales, rotations, harmonics, opacities = _random_inputs()
    path = tmp_path / "scene.ply"
    written = dump_ply(path, means, scales, rotations, harmonics, opacities)
    assert written == _N

    names, data = _read_ply(path)
    num_rest = 3 * (_D_SH - 1)
    assert names == (
        ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(num_rest)]
        + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    )

    assert np.allclose(_field(data, ["x", "y", "z"]), means.numpy(), atol=1e-6)
    assert np.allclose(_field(data, ["nx", "ny", "nz"]), 0.0)
    # Viewer-side activations recover the rendered values.
    loaded_opacity = 1.0 / (1.0 + np.exp(-data["opacity"]))
    assert np.allclose(loaded_opacity, opacities.numpy(), atol=1e-5)
    loaded_scales = np.exp(_field(data, ["scale_0", "scale_1", "scale_2"]))
    assert np.allclose(loaded_scales, scales.numpy(), rtol=1e-5)
    # WXYZ on disk -> XYZW in memory.
    loaded_quat = _field(data, ["rot_1", "rot_2", "rot_3", "rot_0"])
    assert np.allclose(loaded_quat, rotations.numpy(), atol=1e-6)
    # SH: DC verbatim, rest channel-major.
    assert np.allclose(_field(data, ["f_dc_0", "f_dc_1", "f_dc_2"]), harmonics[:, :, 0].numpy(), atol=1e-6)
    loaded_rest = _field(data, [f"f_rest_{i}" for i in range(num_rest)]).reshape(_N, 3, _D_SH - 1)
    assert np.allclose(loaded_rest, harmonics[:, :, 1:].numpy(), atol=1e-6)


def test_quaternion_order_is_wxyz_on_disk(tmp_path):
    means, scales, _, harmonics, opacities = _random_inputs()
    identity_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(_N, 1)
    path = tmp_path / "identity.ply"
    dump_ply(path, means, scales, identity_xyzw, harmonics, opacities)
    _, data = _read_ply(path)
    assert np.allclose(data["rot_0"], 1.0)  # w first
    assert np.allclose(data["rot_1"], 0.0)


def test_sh_dc_only_drops_rest_bands(tmp_path):
    means, scales, rotations, harmonics, opacities = _random_inputs()
    path = tmp_path / "dc_only.ply"
    dump_ply(path, means, scales, rotations, harmonics, opacities, sh_dc_only=True)
    names, data = _read_ply(path)
    assert not any(n.startswith("f_rest") for n in names)
    assert data.shape[0] == _N


def test_mask_subsets_written_gaussians(tmp_path):
    means, scales, rotations, harmonics, opacities = _random_inputs()
    mask = torch.zeros(_N, dtype=torch.bool)
    mask[::3] = True
    path = tmp_path / "masked.ply"
    written = dump_ply(path, means, scales, rotations, harmonics, opacities, mask=mask)
    assert written == int(mask.sum())
    _, data = _read_ply(path)
    assert data.shape[0] == written
    assert np.allclose(_field(data, ["x", "y", "z"]), means[mask].numpy(), atol=1e-6)


def test_shift_and_scale_recenters(tmp_path):
    means, scales, rotations, harmonics, opacities = _random_inputs()
    means = means + 100.0  # far off-center scene
    path = tmp_path / "framed.ply"
    dump_ply(path, means, scales, rotations, harmonics, opacities, shift_and_scale=True)
    _, data = _read_ply(path)
    centers = _field(data, ["x", "y", "z"])
    assert np.abs(np.median(centers, axis=0)).max() < 1e-5
    # The scale factor is the max per-axis 95th quantile, so per-axis q95 <= 1.
    assert np.quantile(np.abs(centers), 0.95, axis=0).max() <= 1.0 + 1e-5


def test_zero_scale_stays_finite(tmp_path):
    means, scales, rotations, harmonics, opacities = _random_inputs()
    scales[0] = 0.0  # the MUSA fused-softplus failure mode
    opacities[1] = 0.0
    opacities[2] = 1.0
    path = tmp_path / "edge.ply"
    dump_ply(path, means, scales, rotations, harmonics, opacities)
    _, data = _read_ply(path)
    for name in data.dtype.names:
        assert np.isfinite(data[name]).all(), f"non-finite values in {name}"


def test_dump_gaussians_dataclass_and_batched_mask(tmp_path):
    means, scales, rotations, harmonics, opacities = _random_inputs()
    gaussians = Gaussians(
        means=means.unsqueeze(0),
        covariances=build_covariance(scales, rotations).unsqueeze(0),
        harmonics=harmonics.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        scales=scales.unsqueeze(0),
        rotations=rotations.unsqueeze(0),
    )
    mask = torch.ones(1, _N, dtype=torch.bool)
    mask[0, :5] = False
    path = tmp_path / "from_dataclass.ply"
    written = dump_gaussians_ply(path, gaussians, batch_index=0, mask=mask)
    assert written == _N - 5


def test_rejects_bad_harmonics_layout(tmp_path):
    means, scales, rotations, harmonics, opacities = _random_inputs()
    with pytest.raises(ValueError, match="harmonics"):
        dump_ply(tmp_path / "bad.ply", means, scales, rotations, harmonics[:, 0], opacities)


# --- per-frame ply (one file per input view) -------------------------------
_V, _H, _W = 3, 24, 32


def _read_vertex_count(path):
    header = open(path, "rb").read(512).partition(b"end_header")[0].decode("ascii")
    line = next(x for x in header.splitlines() if x.startswith("element vertex "))
    return int(line.split()[-1])


def _decoder_and_map(sh_degree=1):
    decoder = GSDecoder(sh_degree=sh_degree)
    torch.manual_seed(0)
    gs_map = torch.randn(1, _V, _H, _W, decoder.raw_gs_dim)
    points = torch.randn(1, _V, _H, _W, 3)
    return decoder, gs_map, points


def test_dump_per_frame_ply_writes_every_pixel_without_a_mask(tmp_path):
    decoder, gs_map, points = _decoder_and_map()

    counts = dump_per_frame_ply(tmp_path, decoder, gs_map, points)

    assert counts == [_H * _W] * _V
    for frame in range(_V):
        assert _read_vertex_count(tmp_path / f"frame_{frame:04d}.ply") == _H * _W


def test_dump_per_frame_ply_honours_the_pixel_mask(tmp_path):
    """Masked dumps must match the mask exactly -- this count is what tells a
    reader how much of the frame survived the 2D confidence filter."""
    decoder, gs_map, points = _decoder_and_map()
    mask = torch.zeros(1, _V, _H, _W, dtype=torch.bool)
    mask[0, 0, :4] = True   # 4 rows
    mask[0, 1, :7] = True   # 7 rows
    mask[0, 2] = True       # everything

    counts = dump_per_frame_ply(tmp_path, decoder, gs_map, points, pixel_mask=mask)

    assert counts == [4 * _W, 7 * _W, _H * _W]
    assert _read_vertex_count(tmp_path / "frame_0000.ply") == 4 * _W


def test_dump_per_frame_ply_dc_only_drops_the_rest_bands(tmp_path):
    decoder, gs_map, points = _decoder_and_map(sh_degree=1)

    dump_per_frame_ply(tmp_path / "full", decoder, gs_map, points)
    dump_per_frame_ply(tmp_path / "dc", decoder, gs_map, points, sh_dc_only=True)

    full = (tmp_path / "full" / "frame_0000.ply").stat().st_size
    dc = (tmp_path / "dc" / "frame_0000.ply").stat().st_size
    assert dc < full
