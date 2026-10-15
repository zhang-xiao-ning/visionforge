"""Tests for DINOTask.

The interesting bits are the lifecycle: attach builds a frozen teacher
with a fresh centre, step_end runs EMA + centre update, and the
checkpoint round-trip restores everything bit-identically.
"""

import pytest
import torch

from application.models.dino import DINOBackbone
from application.tasks.dino import (
    TEACHER_TEMP_WARMUP_STEPS,
    DINOTask,
)


def _model(head_out: int = 8) -> DINOBackbone:
    return DINOBackbone(
        img_size=32,
        patch_size=4,
        embed_dim=16,
        depth=1,
        num_heads=2,
        head_hidden=32,
        head_out=head_out,
        dropout=0.0,
    )


def _batch(B: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(B, 3, 32, 32), torch.randn(B, 3, 32, 32)


# ---------- attach ----------


def test_attach_builds_frozen_teacher() -> None:
    task = DINOTask()
    student = _model(head_out=8)
    task.attach(student, torch.device("cpu"))

    assert task.teacher is not student
    assert all(not p.requires_grad for p in task.teacher.parameters())
    assert task.teacher.training is False


def test_attach_initialises_center_and_step_count() -> None:
    task = DINOTask()
    task.attach(_model(head_out=8), torch.device("cpu"))

    assert task.center.shape == (8,)
    assert torch.equal(task.center, torch.zeros(8))
    assert task.step_count == 0


def test_teacher_is_copy_of_student_at_attach() -> None:
    task = DINOTask()
    student = _model(head_out=8)
    task.attach(student, torch.device("cpu"))

    for ps, pt in zip(student.parameters(), task.teacher.parameters(), strict=True):
        assert torch.equal(ps, pt)


# ---------- train_step ----------


def test_train_step_returns_scalar_positive() -> None:
    task = DINOTask()
    student = _model(head_out=8)
    task.attach(student, torch.device("cpu"))

    loss = task.train_step(student, _batch(), torch.device("cpu"), torch.float32)
    assert loss.shape == ()
    assert loss.item() > 0


def test_train_step_does_not_touch_teacher_grads() -> None:
    task = DINOTask()
    student = _model(head_out=8)
    task.attach(student, torch.device("cpu"))

    loss = task.train_step(student, _batch(), torch.device("cpu"), torch.float32)
    loss.backward()
    for p in task.teacher.parameters():
        assert p.grad is None


# ---------- step_end ----------
def test_step_end_ema_moves_teacher_toward_student() -> None:
    task = DINOTask()
    student = _model(head_out=8)
    task.attach(student, torch.device("cpu"))
    # step_end expects a prior train_step to populate the cached
    # teacher logits used by the centre update.
    task.train_step(student, _batch(), torch.device("cpu"), torch.float32)

    with torch.no_grad():
        for p in student.parameters():
            p.add_(1.0)

    teacher_before = {k: v.clone() for k, v in task.teacher.state_dict().items()}
    task.step_end(student)

    for k, v_after in task.teacher.state_dict().items():
        assert not torch.equal(teacher_before[k], v_after)


def test_step_end_updates_center_and_step_count() -> None:
    task = DINOTask()
    student = _model(head_out=8)
    task.attach(student, torch.device("cpu"))
    task.train_step(student, _batch(), torch.device("cpu"), torch.float32)

    center_before = task.center.clone()
    task.step_end(student)

    assert task.step_count == 1
    assert not torch.equal(center_before, task.center)


def test_teacher_temp_warms_up_linearly() -> None:
    task = DINOTask()
    task.attach(_model(head_out=8), torch.device("cpu"))

    assert task._teacher_temp() == pytest.approx(0.04)
    task.step_count = TEACHER_TEMP_WARMUP_STEPS // 2
    assert 0.04 < task._teacher_temp() < 0.07
    task.step_count = TEACHER_TEMP_WARMUP_STEPS
    assert task._teacher_temp() == pytest.approx(0.07)
    task.step_count = 1000
    assert task._teacher_temp() == pytest.approx(0.07)


# ---------- checkpoint round-trip ----------


def test_state_dict_round_trip_preserves_teacher() -> None:
    task = DINOTask()
    student = _model(head_out=8)
    task.attach(student, torch.device("cpu"))
    task.train_step(student, _batch(), torch.device("cpu"), torch.float32)
    task.step_end(student)

    snapshot = task.state_dict()

    task2 = DINOTask()
    task2.attach(_model(head_out=8), torch.device("cpu"))
    task2.load_state_dict(snapshot)

    assert torch.equal(task.center, task2.center)
    assert task.step_count == task2.step_count
    for (k1, v1), (k2, v2) in zip(
        task.teacher.state_dict().items(),
        task2.teacher.state_dict().items(),
        strict=True,
    ):
        assert k1 == k2
        assert torch.equal(v1, v2)
