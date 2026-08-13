"""CPU unit tests for the AnySplat GSDPTHead / GSDecoder port."""

import pytest
import torch

from vggt_omega.models.heads.gaussian_head import (
    GSDecoder,
    GSDPTHead,
    stable_softplus,
    quaternion_to_matrix,
)
from vggt_omega.utils.geometry import invert_transform_points, transform_points, unproject_depth

_B, _S, _H, _W = 1, 2, 16, 16
_PATCH = 4
_DIM = 8
_SH_DEGREE = 1  # raw_gs_dim = 1 + 7 + 3 * 4 = 20
_TOKEN_START = 1  # one fake special token prepended


def _tiny_head(**overrides):
    kwargs = dict(
        dim_in=_DIM,
        patch_size=_PATCH,
        sh_degree=_SH_DEGREE,
        features=8,
        out_channels=[8, 8, 8, 8],
        intermediate_layer_idx=[0, 1, 2, 3],
    )
    kwargs.update(overrides)
    return GSDPTHead(**kwargs)


def _tiny_inputs(dtype=torch.float32):
    torch.manual_seed(0)
    num_patches = (_H // _PATCH) * (_W // _PATCH)
    tokens = [torch.randn(_B, _S, _TOKEN_START + num_patches, _DIM, dtype=dtype) for _ in range(4)]
    images = torch.rand(_B, _S, 3, _H, _W)
    return tokens, images


def _identity_cameras():
    extrinsics = torch.eye(4)[:3].expand(_B, _S, 3, 4).contiguous()
    intrinsics = torch.tensor([[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]])
    return extrinsics, intrinsics.expand(_B, _S, 3, 3).contiguous()


def test_head_output_shapes():
    head = _tiny_head()
    tokens, images = _tiny_inputs()
    gs_map, gs_conf = head(tokens, images, _TOKEN_START)

    raw_gs_dim = 1 + 7 + 3 * (_SH_DEGREE + 1) ** 2
    assert head.raw_gs_dim == raw_gs_dim
    assert gs_map.shape == (_B, _S, _H, _W, raw_gs_dim)
    assert gs_conf.shape == (_B, _S, _H, _W)
    assert gs_map.dtype == torch.float32
    assert gs_conf.dtype == torch.float32
    assert torch.isfinite(gs_map).all()
    assert torch.isfinite(gs_conf).all()


def test_head_chunked_matches_unchunked():
    head = _tiny_head().eval()
    tokens, images = _tiny_inputs()
    with torch.no_grad():
        gs_map_full, gs_conf_full = head(tokens, images, _TOKEN_START, frames_chunk_size=None)
        gs_map_chunk, gs_conf_chunk = head(tokens, images, _TOKEN_START, frames_chunk_size=1)
    assert torch.allclose(gs_map_full, gs_map_chunk, atol=1e-6)
    assert torch.allclose(gs_conf_full, gs_conf_chunk, atol=1e-6)


def test_head_accepts_bf16_tokens():
    head = _tiny_head()
    tokens, images = _tiny_inputs(dtype=torch.bfloat16)
    gs_map, gs_conf = head(tokens, images, _TOKEN_START)
    assert gs_map.dtype == torch.float32
    assert gs_conf.dtype == torch.float32


def test_head_rejects_missing_cached_layer():
    head = _tiny_head()
    tokens, images = _tiny_inputs()
    tokens[2] = None
    with pytest.raises(ValueError, match="did not cache layer"):
        head(tokens, images, _TOKEN_START)


def test_head_state_dict_matches_anysplat_naming():
    """Sentinel keys/shapes of the shipped AnySplat config (sh_degree=4 -> 84ch)."""
    head = GSDPTHead(dim_in=2048, sh_degree=4)
    state = head.state_dict()
    assert state["projects.0.weight"].shape == (256, 2048, 1, 1)
    assert state["scratch.output_conv1.weight"].shape == (128, 256, 3, 3)
    assert state["input_merger.0.weight"].shape == (128, 3, 7, 7)
    assert state["scratch.output_conv2.2.weight"].shape == (84, 128, 1, 1)
    assert "scratch.refinenet1.resConfUnit2.conv1.weight" in state
    assert "scratch.refinenet4.resConfUnit2.conv1.weight" in state
    assert "scratch.refinenet4.resConfUnit1.conv1.weight" not in state  # has_residual=False


def test_decode_shapes_and_ranges():
    torch.manual_seed(0)
    decoder = GSDecoder(sh_degree=_SH_DEGREE)
    n = _S * _H * _W
    raw = torch.randn(_B, n, decoder.raw_gs_dim)
    means_in = torch.randn(_B, n, 3)

    gaussians = decoder.decode(raw, means_in)

    assert gaussians.means.shape == (_B, n, 3)
    assert gaussians.covariances.shape == (_B, n, 3, 3)
    assert gaussians.harmonics.shape == (_B, n, 3, decoder.d_sh)
    assert gaussians.opacities.shape == (_B, n)
    assert gaussians.scales.shape == (_B, n, 3)
    assert gaussians.rotations.shape == (_B, n, 4)

    assert gaussians.opacities.min() >= 0 and gaussians.opacities.max() <= 1
    assert gaussians.scales.min() > 0 and gaussians.scales.max() <= decoder.scale_max
    assert torch.allclose(gaussians.rotations.norm(dim=-1), torch.ones(_B, n), atol=1e-5)
    # Covariances are symmetric PSD by construction.
    assert torch.allclose(gaussians.covariances, gaussians.covariances.transpose(-1, -2), atol=1e-9)


def test_decode_semantics():
    torch.manual_seed(0)
    decoder = GSDecoder(sh_degree=_SH_DEGREE)
    n = _S * _H * _W
    raw = torch.randn(_B, n, decoder.raw_gs_dim)
    depth = torch.rand(_B, _S, _H, _W) + 0.5
    _, intrinsics = _identity_cameras()

    # Unfused per-pixel decode: flattened gs_map on unprojected points (the
    # documented escape hatch around the fused forward).
    means_in = unproject_depth(depth, intrinsics).reshape(_B, n, 3)
    gaussians = decoder.decode(raw, means_in)

    assert torch.allclose(gaussians.means, means_in, atol=1e-6)

    # Default opacity mapping (initial=final=0) is the identity on the density.
    assert torch.allclose(gaussians.opacities, raw[..., 0].sigmoid(), atol=1e-6)

    # Scales follow the 1e-3 * softplus decode.
    expected_scales = 1e-3 * stable_softplus(raw[..., 1:4])
    assert torch.allclose(gaussians.scales, expected_scales, atol=1e-9)

    # SH degree-1 coefficients are damped by 0.1 * 0.25.
    raw_sh = raw[..., 8:].reshape(_B, n, 3, decoder.d_sh)
    assert torch.allclose(gaussians.harmonics[..., 0], raw_sh[..., 0], atol=1e-7)
    assert torch.allclose(gaussians.harmonics[..., 1:], raw_sh[..., 1:] * 0.025, atol=1e-7)


def test_reference_frame_unprojection_round_trip():
    """The geometry pair used by ``GSDecoder.forward``: camera-frame
    unprojection mapped into the reference frame inverts back exactly."""
    torch.manual_seed(1)
    depth = torch.rand(_B, _S, _H, _W) + 0.5
    _, intrinsics = _identity_cameras()

    rotation = quaternion_to_matrix(torch.nn.functional.normalize(torch.randn(_B, _S, 4), dim=-1))
    translation = torch.randn(_B, _S, 3, 1)
    extrinsics = torch.cat([rotation, translation], dim=-1)

    camera_points = unproject_depth(depth, intrinsics)
    reference_points = invert_transform_points(extrinsics, camera_points)
    assert torch.allclose(transform_points(extrinsics, reference_points), camera_points, atol=1e-5)


def test_decoder_opacity_warm_up():
    decoder = GSDecoder(sh_degree=_SH_DEGREE, opacity_initial=-2.0, opacity_final=0.0, opacity_warm_up=10)
    pdf = torch.linspace(0.0, 1.0, 11)

    cold = decoder.map_pdf_to_opacity(pdf, global_step=0)
    warm = decoder.map_pdf_to_opacity(pdf, global_step=10)

    assert torch.allclose(warm, pdf, atol=1e-6)  # exponent 2^0 = 1 -> identity
    assert not torch.allclose(cold, pdf)
    # The mapping fixes the endpoints and stays monotonic at any warmth.
    assert torch.allclose(cold[0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(cold[-1], torch.tensor(1.0), atol=1e-6)
    assert (cold.diff() >= 0).all()
    # None = fully warmed up.
    assert torch.allclose(decoder.map_pdf_to_opacity(pdf), pdf, atol=1e-6)


def test_decode_rejects_channel_mismatch():
    decoder = GSDecoder(sh_degree=_SH_DEGREE)
    raw = torch.randn(_B, 8, decoder.raw_gs_dim + 1)
    with pytest.raises(ValueError, match="channels"):
        decoder.decode(raw, torch.randn(_B, 8, 3))


def test_head_decode_end_to_end():
    head = _tiny_head()
    decoder = GSDecoder(sh_degree=_SH_DEGREE)
    tokens, images = _tiny_inputs()
    _, intrinsics = _identity_cameras()

    gs_map, _ = head(tokens, images, _TOKEN_START)
    depth = torch.rand(_B, _S, _H, _W) + 0.5
    n = _S * _H * _W
    means_in = unproject_depth(depth, intrinsics).reshape(_B, n, 3)
    gaussians = decoder.decode(gs_map.reshape(_B, n, -1), means_in, global_step=100)

    for field in (gaussians.means, gaussians.covariances, gaussians.harmonics, gaussians.opacities):
        assert torch.isfinite(field).all()
        assert field.dtype == torch.float32


# --- opacity warm-up ----------------------------------------------------------
def test_opacity_mapping_is_the_identity_at_the_defaults():
    """initial == final == 0 must leave pdf untouched, or every existing config
    silently changes behaviour."""
    decoder = GSDecoder(sh_degree=1)
    pdf = torch.rand(500)
    torch.testing.assert_close(decoder.map_pdf_to_opacity(pdf, global_step=0), pdf, atol=1e-6, rtol=1e-6)


def test_positive_opacity_x_pushes_opacity_up():
    """The point of the warm-up: force commitment to opaque surfaces early,
    instead of letting the photometric loss fade everything out."""
    decoder = GSDecoder(sh_degree=1, opacity_initial=2.0, opacity_final=0.0, opacity_warm_up=100)
    pdf = torch.full((256,), 0.5)
    assert float(decoder.map_pdf_to_opacity(pdf, global_step=0).mean()) > 0.85


def test_opacity_warm_up_decays_back_to_the_identity():
    decoder = GSDecoder(sh_degree=1, opacity_initial=2.0, opacity_final=0.0, opacity_warm_up=100)
    pdf = torch.rand(500)
    early = decoder.map_pdf_to_opacity(pdf, global_step=0).mean()
    late = decoder.map_pdf_to_opacity(pdf, global_step=100).mean()
    assert float(early) > float(late)
    torch.testing.assert_close(decoder.map_pdf_to_opacity(pdf, global_step=100), pdf, atol=1e-6, rtol=1e-6)
    # Past the warm-up the schedule must stay clamped, not keep extrapolating.
    torch.testing.assert_close(
        decoder.map_pdf_to_opacity(pdf, global_step=10_000), pdf, atol=1e-6, rtol=1e-6
    )


def test_model_plumbs_the_opacity_schedule_into_the_decoder():
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(
        embed_dim=64, enable_3dgs=True, gs_sh_degree=1,
        gs_opacity_initial=2.0, gs_opacity_final=0.0, gs_opacity_warm_up=500,
    )
    assert (model.gs_decoder.opacity_initial, model.gs_decoder.opacity_warm_up) == (2.0, 500)
