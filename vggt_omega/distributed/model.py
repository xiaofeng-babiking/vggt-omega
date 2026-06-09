"""Context-parallel VGGT-Omega: swaps in CP aggregator + CP camera head.

The DenseHead is reused unchanged (per-frame, no comms). The parameter tree is
identical to VGGTOmega, so the released checkpoint loads with strict=True.
"""
import torch

from vggt_omega.models import VGGTOmega

from .aggregator import ContextParallelAggregator
from .camera_head import ContextParallelCameraHead


class ContextParallelVGGTOmega(VGGTOmega):
    def __init__(
        self,
        cp_group,
        strategy,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
    ) -> None:
        super().__init__(patch_size, embed_dim, enable_camera, enable_depth, enable_alignment)

        self.aggregator = ContextParallelAggregator(patch_size=patch_size, embed_dim=embed_dim)
        self.aggregator.cp_group = cp_group
        self.aggregator.strategy = strategy

        if self.camera_head is not None:
            self.camera_head = ContextParallelCameraHead(dim_in=2 * embed_dim)
            self.camera_head.cp_group = cp_group
            self.camera_head.strategy = strategy
        # text_alignment_head, if enabled, also mixes across frames; out of scope
        # (disabled for camera/depth inference). Guard against silent wrong results:
        if self.text_alignment_head is not None:
            raise NotImplementedError("Context-parallel text-alignment head is not implemented")


def build_cp_model(checkpoint_path: str, cp_group, strategy, device) -> ContextParallelVGGTOmega:
    """Build a CP model and load the released checkpoint into it."""
    model = ContextParallelVGGTOmega(cp_group=cp_group, strategy=strategy).to(device).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    return model
