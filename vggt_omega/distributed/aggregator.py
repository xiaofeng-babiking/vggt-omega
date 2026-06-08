"""Frame-sharded (context-parallel) aggregator.

Each rank holds a contiguous shard of frames. Frame blocks run locally; the
inter-frame "global" and "register" attention is computed across all ranks via a
DistributedAttention strategy. Set `.cp_group` and `.strategy` before forward.
"""
import torch
import torch.distributed as dist

from vggt_omega.models.aggregator import Aggregator
from vggt_omega.models.aggregator import slice_expand_and_flatten as _base_slice

from .block import distributed_block_forward
from .process_group import all_gather_ints


def slice_expand_and_flatten_cp(token_tensor, batch_size, num_frames, global_offset):
    """Offset-aware token assignment.

    `token_tensor` is (1, 2, K, D): [:, 0] is the special first-frame token, [:, 1]
    is the other-frame token. Only the rank owning GLOBAL frame 0 applies the
    first-frame token (to its local frame 0); all other ranks use the other-frame
    token for every local frame.
    """
    if num_frames == 0:
        # Empty shard (N < world_size): content is irrelevant, only the (0, K, D)
        # shape matters so the layer loop stays rank-symmetric.
        return token_tensor.new_zeros(0, token_tensor.shape[2], token_tensor.shape[3])
    if global_offset == 0:
        return _base_slice(token_tensor, batch_size, num_frames)
    other = token_tensor[:, 1:].expand(batch_size, num_frames, *token_tensor.shape[2:])
    return other.reshape(batch_size * num_frames, *other.shape[2:])


class ContextParallelAggregator(Aggregator):
    cp_group = None
    strategy = None

    def forward(self, images: torch.Tensor):
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        counts = all_gather_ints(num_frames, self.cp_group, device=images.device)
        rank = dist.get_rank(self.cp_group)
        global_offset = sum(counts[:rank])

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten_cp(self.camera_token, batch_size, num_frames, global_offset)
        register_token = slice_expand_and_flatten_cp(self.register_token, batch_size, num_frames, global_offset)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens, batch_size, num_frames, num_tokens, embed_dim, block_idx, frame_rope
            )
            tokens = self._run_inter_frame_attention_block(
                tokens, batch_size, num_frames, num_tokens, embed_dim,
                block_idx, self.inter_frame_attention_types[block_idx],
            )
            if block_idx in self.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, self.patch_token_start

    def _run_inter_frame_attention_block(
        self, tokens, batch_size, num_frames, num_tokens, embed_dim, block_idx, attention_type
    ):
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
        block = self.inter_frame_blocks[block_idx]

        if attention_type == "global":
            x = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            x = distributed_block_forward(block, x, self.cp_group, self.strategy, rope=None)
            return x.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        pts = self.patch_token_start
        cam_reg = tokens[:, :, :pts].reshape(batch_size, num_frames * pts, embed_dim)
        patch = tokens[:, :, pts:]  # (B, N, num_tokens-pts, D) -- stays local, unchanged
        cam_reg = distributed_block_forward(block, cam_reg, self.cp_group, self.strategy, rope=None)
        cam_reg = cam_reg.view(batch_size, num_frames, pts, embed_dim)
        return torch.cat([cam_reg, patch], dim=2)
