"""Tests for Task base class."""

import torch
import torch.nn as nn

from application.tasks import ClassificationTask
from application.tasks.captioning import CaptioningTask
from framework.interfaces import Task


def test_task_default_from_data_returns_instance() -> None:
    task = ClassificationTask.from_data(bundle=object())  # type: ignore[arg-type]
    assert isinstance(task, ClassificationTask)


def test_task_from_data_inherited() -> None:
    assert "from_data" not in ClassificationTask.__dict__
    assert "from_data" not in CaptioningTask.__dict__


def test_task_has_no_stages_attribute() -> None:
    assert not hasattr(Task, "stages")


def test_task_has_no_eval_step() -> None:
    assert not hasattr(Task, "eval_step")


def test_task_has_no_primary_metric() -> None:
    assert not hasattr(Task, "primary_metric")


def test_task_has_no_higher_is_better() -> None:
    assert not hasattr(Task, "higher_is_better")


def test_attach_default_is_noop() -> None:
    task = ClassificationTask()
    task.attach(nn.Linear(4, 2), torch.device("cpu"))


def test_step_end_default_is_noop() -> None:
    task = ClassificationTask()
    task.step_end(nn.Linear(4, 2))


def test_state_dict_default_is_empty() -> None:
    assert ClassificationTask().state_dict() == {}


def test_load_state_dict_default_is_noop() -> None:
    ClassificationTask().load_state_dict({"anything": 1})
