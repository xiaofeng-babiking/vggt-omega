import torch

PLANE_Z = 3.0
SCENE_H = 32
SCENE_W = 32
SCENE_FOCAL = 35.0


def _intrinsics_for_scene(B=1, S=3, H=SCENE_H, W=SCENE_W):
    K = torch.zeros(B, S, 3, 3)
    K[..., 0, 0] = SCENE_FOCAL
    K[..., 1, 1] = SCENE_FOCAL
    K[..., 0, 2] = W / 2.0
    K[..., 1, 2] = H / 2.0
    K[..., 2, 2] = 1.0
    return K


def _axis_angle_to_mat(axis, angle):
    axis = axis / axis.norm(dim=-1, keepdim=True)
    kx, ky, kz = axis.unbind(-1)
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], dim=-1
    ).reshape(axis.shape[:-1] + (3, 3))
    eye = torch.eye(3).expand_as(K)
    sin = torch.sin(angle)[..., None, None]
    cos = torch.cos(angle)[..., None, None]
    return eye + sin * K + (1.0 - cos) * (K @ K)


def _random_consistent_scene(B=1, S=3, H=SCENE_H, W=SCENE_W, seed=0, drop_frac=0.05):
    """Random small-rotation w2c cameras viewing the world plane z=PLANE_Z.

    Depth and world_points follow the dataset GT convention exactly
    (datasets/dataset_util.py depth_to_world_coords_points): pixel grid is raw
    arange (no half-pixel offset), cam = depth * K^-1 [u, v, 1],
    world = R^T (cam - t).

    Returns (extrinsics (B,S,3,4), depths (B,S,H,W), world_points (B,S,H,W,3),
    point_masks (B,S,H,W) bool), all fp32.
    """
    g = torch.Generator().manual_seed(seed)
    K = _intrinsics_for_scene(B, S, H, W)

    axis = torch.randn(B, S, 3, generator=g)
    angle = torch.rand(B, S, generator=g) * 0.25
    R = _axis_angle_to_mat(axis, angle)
    t = torch.randn(B, S, 3, generator=g) * 0.2
    extrinsics = torch.cat([R, t[..., None]], dim=-1)

    vs, us = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    fx, fy = K[..., 0, 0], K[..., 1, 1]
    cx, cy = K[..., 0, 2], K[..., 1, 2]
    dirs = torch.stack(
        [
            (us - cx[..., None, None]) / fx[..., None, None],
            (vs - cy[..., None, None]) / fy[..., None, None],
            torch.ones(B, S, H, W),
        ],
        dim=-1,
    )
    ray_dir = torch.einsum("bsji,bshwj->bshwi", R, dirs)
    cam_center = -torch.einsum("bsji,bsj->bsi", R, t)

    depths = (PLANE_Z - cam_center[..., 2, None, None]) / ray_dir[..., 2]
    world_points = cam_center[:, :, None, None] + depths[..., None] * ray_dir

    point_masks = depths > 1e-8
    if drop_frac > 0:
        keep = torch.rand(B, S, H, W, generator=g) >= drop_frac
        point_masks = point_masks & keep
    depths = torch.where(point_masks, depths, torch.zeros_like(depths))
    world_points = torch.where(
        point_masks[..., None], world_points, torch.zeros_like(world_points)
    )
    return extrinsics, depths, world_points, point_masks
