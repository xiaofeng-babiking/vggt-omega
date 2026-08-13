# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dump 3D gaussians to a SuperSplat-compatible INRIA-3DGS ``.ply``.

Ported from AnySplat@6dced92 ``src/model/ply_export.py`` (``export_ply``),
with the file writer following the streaming zero-dependency approach of the
DGGT port (``vggt_omega/rendering/gaussian_ply.py`` on dev/bbk/dggt-gs-head)
instead of AnySplat's ``plyfile`` + ``scipy`` stack.

Format: standard INRIA-3DGS ``.ply``, ``binary_little_endian``, one ``vertex``
element whose float32 properties are, in order:

    x, y, z          gaussian center (world units)
    nx, ny, nz       normals — zeros (INRIA placeholder; viewers ignore them)
    f_dc_0..2        SH DC color coefficients (written verbatim)
    f_rest_*         higher-order SH coefficients, channel-major
                     (all R coeffs, then G, then B — absent with sh_dc_only)
    opacity          logit (inverse sigmoid) of the activated opacity
    scale_0..2       log of the activated per-axis scale
    rot_0..3         normalized quaternion, WXYZ (rot_0 is w)

Loadable in SuperSplat, antimatter15/splat, gsplat / nerfstudio viewers, etc.

Inputs arrive in *activated* space — the exact ``Gaussians`` fields consumed
by ``gaussian_splat.render.render_by_gsplat`` — and the standard viewer
re-applies ``sigmoid(opacity)`` / ``exp(scale)`` / SH evaluation on load, so
the ``.ply`` reproduces what was rendered.

Deliberate deviations from the AnySplat source:
  - opacity is stored as ``logit(opacity)`` per the INRIA convention.
    AnySplat's exporter writes the activated opacity raw, so viewers
    re-sigmoid it and every splat lands in (0.5, 0.73) — washed out.
  - quaternions reorder XYZW -> WXYZ with plain indexing; AnySplat routed
    them through a scipy ``from_quat``/``as_quat`` round trip (a no-op)
    before the same reorder.
  - scales are clamped to >= 1e-9 before ``log`` so exact zeros (the MUSA
    fused-softplus bug that motivated ``stable_softplus``, AnySplat
    issues/AnySplat-007.yaml) can never write ``-inf``.
  - ``sh_dc_only`` defaults to False (AnySplat defaulted True to save
    memory on sh_degree=4 scenes); an optional boolean ``mask`` subsets the
    dumped gaussians (e.g. ``GSDecoderOutput.valid`` to drop padded slots).
  - vertices stream to disk in chunks, so multi-million-splat scenes never
    materialize the full interleaved buffer.
"""

from pathlib import Path

import numpy as np
import torch

from vggt_omega.models.heads import Gaussians


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().float().cpu().numpy())


def _logit(alpha: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    alpha = np.clip(alpha, eps, 1.0 - eps)
    return np.log(alpha / (1.0 - alpha))


def _ply_properties(num_rest: int) -> tuple[str, ...]:
    return (
        ("x", "y", "z")
        + ("nx", "ny", "nz")
        + ("f_dc_0", "f_dc_1", "f_dc_2")
        + tuple(f"f_rest_{i}" for i in range(num_rest))
        + ("opacity",)
        + ("scale_0", "scale_1", "scale_2")
        + ("rot_0", "rot_1", "rot_2", "rot_3")
    )


def _ply_header(num_vertices: int, num_rest: int) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {int(num_vertices)}"]
    lines += [f"property float {p}" for p in _ply_properties(num_rest)]
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def dump_ply(
    path: str | Path,
    means: torch.Tensor,
    scales: torch.Tensor,
    rotations_xyzw: torch.Tensor,
    harmonics: torch.Tensor,
    opacities: torch.Tensor,
    sh_dc_only: bool = False,
    shift_and_scale: bool = False,
    mask: torch.Tensor | None = None,
    chunk_size: int = 1_000_000,
) -> int:
    """Write one gaussian set to a SuperSplat-compatible ``.ply``.

    Args:
        path: output ``.ply`` path (parent directories are created).
        means: (N, 3) centers, world units.
        scales: (N, 3) activated per-axis scales (> 0), world units.
        rotations_xyzw: (N, 4) quaternions, XYZW (scipy order, as carried by
            ``heads.Gaussians``); stored as WXYZ per the PLY convention.
        harmonics: (N, 3, d_sh) SH coefficients, DC first.
        opacities: (N,) activated opacities in [0, 1].
        sh_dc_only: drop the ``f_rest_*`` bands (smaller file, view-independent
            color only).
        shift_and_scale: recenter on the median gaussian and rescale so ~95%
            of centers fall in [-1, 1] (AnySplat's viewer-framing option).
        mask: optional (N,) bool selection of gaussians to write.
        chunk_size: vertices per write, bounding peak memory.

    Returns the number of gaussians written.
    """
    means = _to_numpy(means).reshape(-1, 3)
    scales = _to_numpy(scales).reshape(-1, 3)
    rotations = _to_numpy(rotations_xyzw).reshape(-1, 4)
    harmonics = _to_numpy(harmonics)
    opacities = _to_numpy(opacities).reshape(-1)
    if harmonics.ndim != 3 or harmonics.shape[1] != 3:
        raise ValueError(f"harmonics must be (N, 3, d_sh), got {harmonics.shape}")

    if mask is not None:
        keep = _to_numpy(mask).reshape(-1).astype(bool)
        means, scales, rotations = means[keep], scales[keep], rotations[keep]
        harmonics, opacities = harmonics[keep], opacities[keep]

    if shift_and_scale:
        means = means - np.median(means, axis=0)
        scale_factor = np.quantile(np.abs(means), 0.95, axis=0).max()
        means = means / scale_factor
        scales = scales / scale_factor

    f_dc_all = harmonics[:, :, 0]
    if sh_dc_only:
        f_rest_all = np.zeros((harmonics.shape[0], 0), dtype=np.float32)
    else:
        # (N, 3, K-1) flattened row-major == channel-major, the INRIA order.
        f_rest_all = harmonics[:, :, 1:].reshape(harmonics.shape[0], -1)
    num_rest = f_rest_all.shape[1]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = means.shape[0]
    with open(path, "wb") as stream:
        stream.write(_ply_header(total, num_rest))
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            count = end - start

            opacity_logit = _logit(opacities[start:end]).reshape(count, 1)
            scale_log = np.log(np.clip(scales[start:end], 1e-9, None))
            quat = rotations[start:end][:, [3, 0, 1, 2]]  # XYZW -> WXYZ
            norm = np.linalg.norm(quat, axis=1, keepdims=True)
            quat = quat / np.where(norm < 1e-12, 1.0, norm)
            normals = np.zeros((count, 3), dtype=np.float32)

            slab = np.concatenate(
                [
                    means[start:end],
                    normals,
                    f_dc_all[start:end],
                    f_rest_all[start:end],
                    opacity_logit,
                    scale_log,
                    quat,
                ],
                axis=1,
            )
            stream.write(slab.astype("<f4", copy=False).tobytes())
    return total


def dump_gaussians_ply(
    path: str | Path,
    gaussians: Gaussians,
    batch_index: int = 0,
    sh_dc_only: bool = False,
    shift_and_scale: bool = False,
    mask: torch.Tensor | None = None,
    chunk_size: int = 1_000_000,
) -> int:
    """Dump one batch element of a ``heads.Gaussians`` set to ``.ply``.

    ``mask`` may be (N,) or (B, N); with (B, N) the ``batch_index`` row is
    used. Returns the number of gaussians written.
    """
    if mask is not None and mask.dim() == 2:
        mask = mask[batch_index]
    return dump_ply(
        path,
        means=gaussians.means[batch_index],
        scales=gaussians.scales[batch_index],
        rotations_xyzw=gaussians.rotations[batch_index],
        harmonics=gaussians.harmonics[batch_index],
        opacities=gaussians.opacities[batch_index],
        sh_dc_only=sh_dc_only,
        shift_and_scale=shift_and_scale,
        mask=mask,
        chunk_size=chunk_size,
    )


def dump_per_frame_ply(
    directory: str | Path,
    decoder: torch.nn.Module,
    gs_map: torch.Tensor,
    points: torch.Tensor,
    pixel_mask: torch.Tensor | None = None,
    sh_dc_only: bool = False,
    batch_index: int = 0,
) -> list[int]:
    """One ``.ply`` per input view, decoded from the UNFUSED per-pixel splats.

    Args:
        directory: output directory (created); files are ``frame_XX.ply``.
        decoder: the model's ``GSDecoder`` -- only ``decode`` is called, so the
            activations exactly match the fused scene's.
        gs_map: (B, S, H, W, raw_gs_dim) raw ``GSDPTHead`` output.
        points: (B, S, H, W, 3) gaussian centers already in the reference
            frame (``invert_transform_points`` of the unprojected depth).
        pixel_mask: optional (B, S, H, W) bool; ``None`` keeps every pixel.
            Pass ``predictions["gaussians_pixel_mask"]`` to keep only the
            pixels that also reached the fused scene.
        sh_dc_only: drop the ``f_rest_*`` bands (~4x smaller files).
        batch_index: which scene of the batch to write.

    Returns the splat count written per view.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    num_frames, raw_dim = gs_map.shape[1], gs_map.shape[-1]
    counts = []
    for frame in range(num_frames):
        raw = gs_map[batch_index, frame].reshape(1, -1, raw_dim)
        centers = points[batch_index, frame].reshape(1, -1, 3)
        mask = None
        if pixel_mask is not None:
            mask = pixel_mask[batch_index, frame].reshape(1, -1)
        counts.append(
            dump_gaussians_ply(
                directory / f"frame_{frame:04d}.ply",
                decoder.decode(raw, centers),
                sh_dc_only=sh_dc_only,
                mask=mask,
            )
        )
    return counts
