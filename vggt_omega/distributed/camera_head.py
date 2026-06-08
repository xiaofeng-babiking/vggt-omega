"""Frame-sharded camera head: the 4-block trunk attends across all frames.

Receives the rank-LOCAL aggregated tokens, so all per-frame work is local; only
the trunk's cross-frame attention is distributed. Set `.cp_group`/`.strategy`.
"""
import torch

from vggt_omega.models.heads.camera_head import CameraHead, _apply_camera_activation

from .block import distributed_block_forward


class ContextParallelCameraHead(CameraHead):
    cp_group = None
    strategy = None

    def forward(self, aggregated_tokens_list, patch_token_start):
        tokens = aggregated_tokens_list[-1]
        if tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which CameraHead needs.")
        batch_size, num_frames, num_tokens, _ = tokens.shape
        if patch_token_start is None:
            raise ValueError("patch_token_start is required for CameraHead")
        if patch_token_start > num_tokens:
            raise ValueError(f"patch_token_start ({patch_token_start}) exceeds token length ({num_tokens})")

        if tokens.dtype != torch.float32:
            tokens = tokens.float()

        cam_reg = tokens[:, :, :patch_token_start]
        cam_reg = self.token_norm(cam_reg)
        cam_reg = cam_reg.reshape(batch_size, num_frames * patch_token_start, -1)
        for block in self.trunk:
            cam_reg = distributed_block_forward(block, cam_reg, self.cp_group, self.strategy, rope=None)
        cam_reg = cam_reg.reshape(batch_size, num_frames, patch_token_start, -1)
        camera_tokens = self.trunk_norm(cam_reg[:, :, 0])
        return _apply_camera_activation(self.camera_branch(camera_tokens))
