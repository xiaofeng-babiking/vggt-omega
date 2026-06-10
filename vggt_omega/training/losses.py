import torch
import torch.nn.functional as F

from vggt_omega.utils.geometry import closed_form_inverse_se3
from vggt_omega.utils.pose_enc import encoding_to_camera, extri_intri_to_pose_encoding


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
    msum = point_masks.sum(dim=(1, 2, 3))
    point_scale = (dist * point_masks).sum(dim=(1, 2, 3)) / msum.clamp(min=1)
    # Samples without any valid point (depth-less vendors like DL3DV) would
    # otherwise clamp to eps and explode the camera GT by 1/eps; fall back to
    # the mean camera-center distance, and to 1.0 (no scaling) if that is also
    # degenerate (single frame / static rig).
    centers = -torch.einsum("bsji,bsj->bsi", new_ext[..., :3], new_ext[..., 3])
    cam_scale = centers.norm(dim=-1).mean(dim=1)
    scale = torch.where(msum > 0, point_scale, cam_scale)
    scale = torch.where(scale > eps, scale, torch.ones_like(scale))
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


def camera_loss(pred_pose_enc, gt_pose_enc):
    """L1 over the 9-D pose encoding, mean over (B, S). GT must come from
    normalized (first-camera-anchored) extrinsics."""
    return (pred_pose_enc - gt_pose_enc).abs().mean()


def _masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def _aleatoric_terms(err, conf, gt_depth, valid, alpha, depth_clamp=1e-3):
    """Shared paper form: c*(1+1/D)*|e| + c*|grad e| - alpha*log c, masked means.

    err (B,S,H,W) or (B,S,H,W,C) = SIGNED residual (the gradient term must see
    sign flips: |grad e| >= |grad |e||), conf (B,S,H,W), gt_depth (B,S,H,W) in
    NORMALIZED units, valid (B,S,H,W) bool.
    """
    if err.dim() == 4:
        err = err.unsqueeze(-1)
    err_abs = err.abs().sum(dim=-1)
    w = 1.0 + 1.0 / gt_depth.clamp(min=depth_clamp)
    data = _masked_mean(conf * w * err_abs, valid)
    gx = (err[:, :, :, 1:] - err[:, :, :, :-1]).abs().sum(dim=-1)
    mx = valid[..., :, 1:] & valid[..., :, :-1]
    gy = (err[:, :, 1:] - err[:, :, :-1]).abs().sum(dim=-1)
    my = valid[..., 1:, :] & valid[..., :-1, :]
    grad = _masked_mean(conf[..., :, :-1] * gx, mx) + _masked_mean(conf[..., :-1, :] * gy, my)
    reg = -alpha * _masked_mean(torch.log(conf.clamp(min=1e-6)), valid)
    return data + grad + reg


def depth_loss(pred_depth, conf, gt_depth, valid, alpha=0.2):
    return _aleatoric_terms(pred_depth - gt_depth, conf, gt_depth, valid, alpha)


def point_loss(pred_depth, conf, pred_pose_enc, gt_points, gt_depth, valid, image_size_hw, alpha=0.2):
    ext, K = encoding_to_camera(pred_pose_enc, image_size_hw, build_intrinsics=True)
    pred_points = unproject_depth(pred_depth, ext, K)
    return _aleatoric_terms(pred_points - gt_points, conf, gt_depth, valid, alpha)


def matching_loss(patch_tokens, tracks, track_vis, track_pos, patch_size, image_size_hw, temperature=1.0):
    """BCE over sigmoid(cosine sim) of last-layer patch tokens at track locations.

    patch_tokens (B,S,P,C) any float dtype (cast to fp32 here); tracks (B,S,T,2) px;
    track_vis (B,S,T) bool (False for ALL negative-track frames by dataset contract);
    track_pos (B,T) bool. Query frame is 0; pairs are (0, i) for i in 1..S-1.
    Positives: vis[0,t] & vis[i,t]. Negatives: ~pos[t] & in-bounds at both frames.
    Returns scalar fp32; zero tensor if S < 2 or no valid pairs.
    """
    B, S, P, C = patch_tokens.shape
    H, W = image_size_hw
    gw = W // patch_size
    if S < 2:
        return patch_tokens.new_zeros((), dtype=torch.float32)
    z = F.normalize(patch_tokens.float(), dim=-1)
    inb = (tracks[..., 0] >= 0) & (tracks[..., 0] < W) & (tracks[..., 1] >= 0) & (tracks[..., 1] < H)
    idx = (
        (tracks[..., 1].clamp(0, H - 1) // patch_size).long() * gw
        + (tracks[..., 0].clamp(0, W - 1) // patch_size).long()
    )
    tok = torch.gather(z, 2, idx.unsqueeze(-1).expand(-1, -1, -1, C))
    sim = (tok[:, :1] * tok).sum(-1)[:, 1:] / temperature
    pos_pair = (track_vis[:, :1] & track_vis[:, 1:]) & track_pos[:, None]
    neg_pair = (inb[:, :1] & inb[:, 1:]) & ~track_pos[:, None]
    loss = patch_tokens.new_zeros((), dtype=torch.float32)
    if pos_pair.any():
        loss = loss + F.binary_cross_entropy_with_logits(
            sim[pos_pair], torch.ones_like(sim[pos_pair])
        )
    if neg_pair.any():
        loss = loss + F.binary_cross_entropy_with_logits(
            sim[neg_pair], torch.zeros_like(sim[neg_pair])
        )
    return loss


class TrainLossComputer:
    """Normalizes GT once, computes all four losses + weighted total.

    weights: dict(camera=5.0, depth=1.0, point=0.5, match=0.1).
    __call__(predictions, batch, image_size_hw) -> dict of scalar tensors:
      total, camera, depth, point, match, gt_scale (detached mean, for logging).
    batch keys: depths (B,S,H,W), extrinsics (B,S,3,4), intrinsics (B,S,3,3),
      world_points (B,S,H,W,3), point_masks (B,S,H,W) bool, optionally
      tracks/track_vis_mask/track_positive_mask.
    predictions keys: pose_enc, depth (B,S,H,W,1) -> squeezed here, depth_conf,
      optionally patch_tokens. The match term is computed only when its weight > 0
      AND track keys AND patch_tokens are present.
    """

    def __init__(self, weights, alpha=0.2, temperature=1.0, patch_size=16):
        self.weights = dict(weights)
        self.alpha = alpha
        self.temperature = temperature
        self.patch_size = patch_size

    def __call__(self, predictions, batch, image_size_hw):
        n_ext, n_dep, n_wp, scale = normalize_gt_into_first_camera(
            batch["extrinsics"], batch["depths"], batch["world_points"], batch["point_masks"]
        )
        valid = batch["point_masks"]
        gt_enc = extri_intri_to_pose_encoding(n_ext, batch["intrinsics"], image_size_hw)
        pred_depth = predictions["depth"].squeeze(-1)
        out = {
            "camera": camera_loss(predictions["pose_enc"], gt_enc),
            "depth": depth_loss(pred_depth, predictions["depth_conf"], n_dep, valid, self.alpha),
            "point": point_loss(
                pred_depth,
                predictions["depth_conf"],
                predictions["pose_enc"],
                n_wp,
                n_dep,
                valid,
                image_size_hw,
                self.alpha,
            ),
            "gt_scale": scale.detach().mean(),
        }
        if self.weights.get("match", 0) > 0 and "patch_tokens" in predictions and "tracks" in batch:
            out["match"] = matching_loss(
                predictions["patch_tokens"],
                batch["tracks"],
                batch["track_vis_mask"],
                batch["track_positive_mask"],
                self.patch_size,
                image_size_hw,
                self.temperature,
            )
        else:
            out["match"] = predictions["pose_enc"].new_zeros(())
        out["total"] = sum(self.weights.get(k, 0.0) * out[k] for k in ("camera", "depth", "point", "match"))
        return out
