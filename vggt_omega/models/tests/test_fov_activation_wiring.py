# vggt_omega/models/tests/test_fov_activation_wiring.py
"""``model.fov_activation`` must reach every camera head that exists.

A config key that is silently dropped is worse than no key: the run reports the
setting it was given and trains with a different one. The context-parallel path
is the specific hazard -- ContextParallelVGGTOmega builds its own camera head
after super().__init__, so it can drop kwargs the plain path honours (it did
exactly that before this test existed).
"""
import pytest
import torch

from vggt_omega.models.heads.camera_head import CameraHead
from vggt_omega.models.vggt_omega import VGGTOmega


@pytest.mark.parametrize("activation", ["softplus", "relu"])
def test_vggt_omega_forwards_fov_activation_to_the_head(activation):
    model = VGGTOmega(embed_dim=64, patch_size=16, fov_activation=activation)
    assert model.camera_head.fov_activation == activation


def test_default_activation_is_escapable():
    """A config that sets nothing must get the fixed behaviour."""
    assert VGGTOmega(embed_dim=64, patch_size=16).camera_head.fov_activation == "softplus"


def test_camera_head_rejects_unknown_activation():
    with pytest.raises(ValueError, match="fov_activation"):
        CameraHead(dim_in=128, fov_activation="swish")


def test_activation_is_not_persisted_in_the_state_dict():
    """It is a plain str, so checkpoints stay loadable across the change in both
    directions -- a run can switch activation without touching its weights."""
    model = VGGTOmega(embed_dim=64, patch_size=16, fov_activation="relu")
    assert not any("fov_activation" in k for k in model.state_dict())
    other = VGGTOmega(embed_dim=64, patch_size=16, fov_activation="softplus")
    other.load_state_dict(model.state_dict())
    assert other.camera_head.fov_activation == "softplus"


@pytest.mark.parametrize("activation", ["softplus", "relu"])
def test_context_parallel_model_keeps_the_activation(activation):
    """The CP head is rebuilt after super().__init__ -- it must carry the setting
    across, or CP runs would silently train on a different parameterization."""
    from vggt_omega.distributed.model import ContextParallelVGGTOmega

    model = ContextParallelVGGTOmega(
        cp_group=None, strategy=None, embed_dim=64, patch_size=16, fov_activation=activation
    )
    assert model.camera_head.fov_activation == activation
