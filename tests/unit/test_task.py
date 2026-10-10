"""Tests for Task base class."""

from framework.interfaces import Task
from tasks.captioning import CaptioningTask
from tasks.classification import ClassificationTask


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
