import torch

from vggt_omega.utils.geometry import closed_form_inverse_se3


def normalize_gt_into_first_camera(extrinsics, depths, world_points, point_masks, eps=1e-6):
    """Re-anchor GT to frame 0's camera and rescale to unit average point distance.

    Args:
        extrinsics: (B, S, 3, 4) world-to-camera OpenCV [R|t].
        depths: (B, S, H, W) GT depth (0 = invalid, <0 = sky).
        world_points: (B, S, H, W, 3) GT points in the (arbitrary) world frame.
        point_masks: (B, S, H, W) bool, valid pixels.

    Returns:
        (extrinsics, depths, world_points, scale (B,)) — new fp32 tensors with
        frame 0 at identity and mean valid-point distance 1. GT-only (never preds).
    """
    B, S = extrinsics.shape[:2]
    first_c2w = closed_form_inverse_se3(extrinsics[:, 0])
    flat = torch.cat(
        [extrinsics, extrinsics.new_tensor([0, 0, 0, 1]).expand(B, S, 1, 4)], dim=2
    )
    new_ext = (flat @ first_c2w[:, None])[:, :, :3]
    R0, t0 = extrinsics[:, 0, :3, :3], extrinsics[:, 0, :3, 3]
    new_wp = torch.einsum("bij,bshwj->bshwi", R0, world_points) + t0[:, None, None, None]
    dist = new_wp.norm(dim=-1)
    msum = point_masks.sum(dim=(1, 2, 3)).clamp(min=1)
    scale = (dist * point_masks).sum(dim=(1, 2, 3)) / msum
    scale = scale.clamp(min=eps)
    sview = scale[:, None, None, None]
    new_ext = new_ext.clone()
    new_ext[..., 3] = new_ext[..., 3] / scale[:, None, None]
    return new_ext, depths / sview, new_wp / sview[..., None], scale


def unproject_depth(depth, extrinsics, intrinsics):
    """Differentiable unprojection: depth (B,S,H,W), w2c extrinsics (B,S,3,4),
    K (B,S,3,3) -> world points (B,S,H,W,3).

    cam = depth * K^-1 [u, v, 1]; world = R^T (cam - t). Torch port of the numpy
    reference (inference.py unproject_depth_map_to_point_map / dataset_util.py
    depth_to_cam_coords_points): raw arange pixel grid, no half-pixel offset.
    """
    B, S, H, W = depth.shape
    vs, us = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    ones = torch.ones_like(us)
    pix = torch.stack([us, vs, ones], dim=-1).reshape(1, 1, H * W, 3)
    Kinv = torch.linalg.inv(intrinsics.float())
    cam = torch.einsum("bsij,bspj->bspi", Kinv, pix.expand(B, S, -1, -1)) * depth.reshape(
        B, S, H * W, 1
    )
    R, t = extrinsics[..., :3], extrinsics[..., 3]
    world = torch.einsum("bsji,bspj->bspi", R, cam - t[:, :, None])
    return world.reshape(B, S, H, W, 3)
