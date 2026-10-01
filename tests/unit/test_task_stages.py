"""Tests for Task.stages / Stage."""

from tasks.base import Stage
from tasks.captioning import CaptioningTask
from tasks.classification import ClassificationTask


def test_stage_defaults() -> None:
    s = Stage(name="align", epochs=1)
    assert s.name == "align"
    assert s.epochs == 1
    assert s.freeze == []


def test_stage_freeze_list() -> None:
    s = Stage(name="align", epochs=1, freeze=["vision", "llm_base"])
    assert s.freeze == ["vision", "llm_base"]


def test_stage_freeze_is_not_shared() -> None:
    """Two Stages must not share the same list instance."""
    a = Stage(name="a", epochs=1)
    b = Stage(name="b", epochs=1)
    a.freeze.append("vision")
    assert b.freeze == []


def test_task_default_stages_is_none() -> None:
    assert ClassificationTask.stages is None
    assert CaptioningTask.stages is None


def test_task_default_from_data_returns_instance() -> None:
    """Default from_data() ignores the bundle and calls cls()."""
    task = ClassificationTask.from_data(bundle=object())  # type: ignore[arg-type]
    assert isinstance(task, ClassificationTask)


def test_task_from_data_inherited() -> None:
    """Both concrete tasks inherit the default, not their own."""
    assert "from_data" not in ClassificationTask.__dict__
    assert "from_data" not in CaptioningTask.__dict__
