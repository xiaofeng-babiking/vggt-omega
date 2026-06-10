import torch

CORE_KEYS = ("images", "depths", "extrinsics", "intrinsics", "world_points", "point_masks", "ids")
OPTIONAL_STACK_KEYS = ("cam_points", "tracks", "track_vis_mask", "track_positive_mask")
LIST_KEYS = ("seq_name", "modalities")


def train_collate(samples):
    """Stack the core training contract; keep shared optional tensors; drop unshared extras.

    All samples in a batch share (S, aspect_ratio) by DynamicBatchSampler construction, so
    core tensors stack without padding. Vendor-specific extras (sky_masks, timestamps, ...)
    differ across vendors and are dropped unless every sample has them.
    """
    out = {}
    for k in CORE_KEYS:
        out[k] = torch.stack([s[k] for s in samples])        # KeyError = contract violation, let it raise
    for k in OPTIONAL_STACK_KEYS:
        if all(k in s for s in samples):
            out[k] = torch.stack([s[k] for s in samples])
    for k in LIST_KEYS:
        out[k] = [s.get(k) for s in samples]
    return out
