"""Unit tests for the frozen parallel teacher.

Everything here runs on CPU with a deliberately tiny aggregator width. The point
is the wiring -- freezing, eval-pinning, buffer materialisation, config
validation -- not numerics, so a 64-dim model exercises the same code paths a 10B
one would.
"""
import pytest
import torch
from omegaconf import OmegaConf

from vggt_omega.models.aggregator import _RESNET_MEAN
from vggt_omega.models.vggt_omega import VGGTOmega
from vggt_omega.training.teacher import (
    Teacher,
    _materialize_nonpersistent_buffers,
    build_teacher,
)
from vggt_omega.training.trainer import init_model_from_scratch

TINY = dict(embed_dim=64, patch_size=16)
IMAGES = (1, 2, 3, 64, 64)  # B, S, C, H, W


def _tiny_model():
    """A forward-capable tiny model. ``init_model_from_scratch`` is not optional:
    LinearKMaskedBias buffers are NaN-by-design, so a bare VGGTOmega returns all
    NaN and every numeric assertion below would compare NaN to NaN."""
    model = VGGTOmega(**TINY)
    init_model_from_scratch(model)
    return model


def _cfg(**teacher):
    return OmegaConf.create({"model": {"embed_dim": 64, "patch_size": 16}, "teacher": teacher})


def test_teacher_holds_no_trainable_parameters():
    teacher = Teacher(_tiny_model())
    assert not any(p.requires_grad for p in teacher.parameters())


def test_train_mode_cannot_unpin_eval():
    """The Trainer calls .train() on what it owns; the teacher must not follow."""
    teacher = Teacher(_tiny_model())
    teacher.train()
    assert not teacher.training
    assert not teacher.model.aggregator.training


def test_forward_returns_detached_fp32_pose_and_depth():
    teacher = Teacher(_tiny_model())
    out = teacher(torch.randn(*IMAGES))
    assert set(out) == {"pose_enc", "depth", "depth_conf"}
    for name, tensor in out.items():
        assert not tensor.requires_grad, name
        assert tensor.grad_fn is None, name
        assert tensor.dtype is torch.float32, name


def test_forward_does_not_build_a_graph_through_an_upstream_input():
    """A teacher wired into a loss must never leak gradient into the student."""
    teacher = Teacher(_tiny_model())
    images = torch.randn(*IMAGES, requires_grad=True)
    out = teacher(images)
    assert out["depth"].grad_fn is None
    assert images.grad is None


def test_materialize_rebuilds_non_persistent_buffers_after_meta_init():
    """persistent=False buffers are absent from the state dict, so to_empty()
    leaves them as uninitialised memory that no load can fix."""
    with torch.device("meta"):
        model = VGGTOmega(**TINY)
    model.to_empty(device="cpu")
    _materialize_nonpersistent_buffers(model)
    assert torch.allclose(
        model.aggregator._resnet_mean.flatten(), torch.tensor(_RESNET_MEAN)
    )
    assert torch.isfinite(model.aggregator._resnet_std).all()


def test_materialize_refuses_an_unknown_non_persistent_buffer():
    """Silently skipping one would corrupt predictions with no error at all."""
    model = _tiny_model()
    model.register_buffer("mystery_constant", torch.empty(3), persistent=False)
    with pytest.raises(RuntimeError, match="mystery_constant"):
        _materialize_nonpersistent_buffers(model)


def test_build_teacher_is_none_when_unconfigured():
    assert build_teacher(_cfg(), torch.device("cpu")) is None
    assert build_teacher(OmegaConf.create({}), torch.device("cpu")) is None


def test_build_teacher_rejects_an_unknown_shard_mode():
    with pytest.raises(ValueError, match="teacher.shard"):
        build_teacher(_cfg(checkpoint="x.pt", shard="magic"), torch.device("cpu"))


def test_build_teacher_requires_a_process_group_to_shard():
    with pytest.raises(RuntimeError, match="process group"):
        build_teacher(_cfg(checkpoint="x.pt", shard="fsdp"), torch.device("cpu"))


def test_build_teacher_replica_round_trips_a_checkpoint(tmp_path):
    reference = _tiny_model().eval()
    path = tmp_path / "teacher.pt"
    torch.save(reference.state_dict(), path)

    teacher = build_teacher(_cfg(checkpoint=str(path), shard="none"), torch.device("cpu"))
    images = torch.randn(*IMAGES)
    with torch.no_grad():
        expected = reference(images)
    got = teacher(images)
    for key in ("pose_enc", "depth", "depth_conf"):
        assert torch.isfinite(got[key]).all(), f"{key} is not finite"
        assert torch.allclose(got[key], expected[key].float(), atol=1e-5), key


def _ema(student, **overrides):
    cfg = _cfg(mode="ema", shard="none", **overrides)
    return build_teacher(cfg, torch.device("cpu"), student=student)


def test_ema_mode_needs_a_student_to_average():
    with pytest.raises(RuntimeError, match="student"):
        build_teacher(_cfg(mode="ema"), torch.device("cpu"))


def test_ema_initialises_from_the_student():
    """With no checkpoint the EMA teacher's starting point IS the student, so the
    two must agree exactly before any update."""
    student = _tiny_model()
    teacher = _ema(student)
    for (name, tp), sp in zip(teacher.model.named_parameters(), student.parameters()):
        assert torch.equal(tp, sp), name


def test_ema_update_interpolates_toward_the_student():
    student = _tiny_model()
    teacher = _ema(student, decay=0.9)
    before = {n: p.clone() for n, p in teacher.model.named_parameters()}
    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)

    teacher.update(student, decay=0.9)

    for name, tp in teacher.model.named_parameters():
        # 0.9 * old + 0.1 * (old + 1) == old + 0.1
        assert torch.allclose(tp, before[name] + 0.1, atol=1e-6), name


def test_ema_update_keeps_the_teacher_frozen_and_in_eval():
    student = _tiny_model()
    teacher = _ema(student)
    teacher.update(student, step=0)
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher.training


def test_ema_rejects_a_student_of_a_different_architecture():
    student = _tiny_model()
    teacher = _ema(student)
    with pytest.raises(RuntimeError, match="architecture"):
        teacher.update(torch.nn.Linear(4, 4))


def test_frozen_teacher_rejects_update():
    """update() is the one mode-gated method: a frozen teacher may be a different
    architecture than the student, so lerping toward it would corrupt it."""
    teacher = Teacher(_tiny_model(), ema=False)
    with pytest.raises(RuntimeError, match="frozen"):
        teacher.update(_tiny_model())


def test_ema_decay_ramps_from_start_to_target():
    teacher = Teacher(_tiny_model(), ema=True, decay=0.99, start_decay=0.9, warmup_steps=10)
    assert teacher.decay_at(0) == pytest.approx(0.909)
    assert teacher.decay_at(9) == pytest.approx(0.99)
    assert teacher.decay_at(1000) == pytest.approx(0.99)
    assert teacher.decay_at(None) == pytest.approx(0.99)


def test_ema_tracks_buffers_exactly_rather_than_averaging_them():
    """Buffers here are constants (ImageNet mean/std, the masked-bias pattern);
    a half-interpolated mask would be silently meaningless."""
    student = _tiny_model()
    teacher = _ema(student, decay=0.5)
    with torch.no_grad():
        student.aggregator._resnet_mean.fill_(0.25)
    teacher.update(student, decay=0.5)
    assert torch.allclose(
        teacher.model.aggregator._resnet_mean, torch.full_like(student.aggregator._resnet_mean, 0.25)
    )


def test_build_teacher_never_builds_heads_the_checkpoint_lacks():
    """enable_alignment must stay off: the released checkpoints carry no
    text_alignment_head, so building one would break a strict load."""
    reference = _tiny_model()
    teacher = Teacher(reference)
    assert teacher.model.text_alignment_head is None
