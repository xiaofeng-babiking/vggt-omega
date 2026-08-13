# vggt_omega/training/tests/test_teacher_frozen.py
"""The teacher is an oracle: it must never train, under any configuration.

The student's camera/depth anchors are an L1 against this model's predictions
(selfsup.OffGridLossComputer._distill), so a teacher that moved would be
regressing the student onto a target that drifts with it -- the anchor would
still read small while both slid together, which is exactly the failure the
anchor exists to catch. Freezing is therefore a hard invariant of the Teacher
wrapper, not a configuration option, and these tests pin it so.

Contrast with the STUDENT, whose freezing is entirely config-driven
(``model.freeze_backbone``, ``selfsup.mode``) and defaults to fully trainable.
"""
import torch
import torch.nn as nn

from vggt_omega.training.teacher import Teacher


def _tiny():
    return nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))


def test_no_teacher_parameter_requires_grad():
    model = _tiny()
    for p in model.parameters():
        p.requires_grad_(True)  # hand it a fully trainable model
    teacher = Teacher(model)
    assert list(teacher.parameters()), "sanity: the teacher should own parameters"
    assert not any(p.requires_grad for p in teacher.parameters())


def test_teacher_is_in_eval_mode():
    """Not cosmetic: a teacher in train mode updates norm statistics every step
    and stops being a fixed reference even with requires_grad off."""
    teacher = Teacher(_tiny())
    assert not teacher.training
    assert not teacher.model.training


def test_train_mode_cannot_unpin_the_teacher():
    """The Trainer calls .train() on what it owns each epoch; the teacher must
    refuse to follow the student into train mode."""
    teacher = Teacher(_tiny())
    teacher.train()
    assert not teacher.training
    teacher.train(True)
    assert not teacher.training
    assert not any(p.requires_grad for p in teacher.parameters())


def test_teacher_output_carries_no_gradient():
    """The end-to-end property: nothing downstream of the teacher can backprop
    into it, so its predictions are constants in the student's graph."""
    teacher = Teacher(_tiny())
    out = teacher.model(torch.randn(2, 4))
    assert not out.requires_grad


def test_ema_teacher_is_still_gradient_frozen():
    """EMA mode moves the weights by averaging, never by gradient -- the
    requires_grad invariant is identical."""
    teacher = Teacher(_tiny(), ema=True, decay=0.99)
    assert not any(p.requires_grad for p in teacher.parameters())
    assert not teacher.training
