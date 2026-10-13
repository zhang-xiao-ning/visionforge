"""Runner integration for the Task lifecycle (attach / step_end / state_dict).

Defaults are no-ops (covered in test_task.py). Here we assert the runner
calls them at the right moments and checkpoints carry task state.
"""

from pathlib import Path

import pytest
import torch

from fakes import TrackableTask, make_classification_builder
from framework.experiment.runner import ExperimentRunner
from framework.experiment.spec import Experiment, TrainConfig
from framework.metrics import Accuracy
from framework.registry import DATASETS, EXPERIMENTS
from framework.training import SingleDeviceStrategy


@pytest.fixture
def spy_runner(monkeypatch, tmp_path: Path):
    exp = Experiment(
        name="spy_exp",
        model=__import__("fakes").TinyModel,
        task=TrackableTask,
        data="fake",
        metrics=[Accuracy],
        primary_metric="acc",
        category="classification",
    )
    monkeypatch.setitem(EXPERIMENTS, "spy_exp", exp)
    monkeypatch.setitem(DATASETS, "fake", make_classification_builder(shape=(4,), num_classes=2))

    def make(accum_steps: int = 1) -> ExperimentRunner:
        return ExperimentRunner(
            experiment=EXPERIMENTS["spy_exp"],
            config=TrainConfig(epochs=1, accum_steps=accum_steps),
            strategy=SingleDeviceStrategy(),
            outputs_dir=tmp_path / "outputs",
            checkpoints_dir=tmp_path / "checkpoints",
        )

    return make


def test_attach_called_once_during_init(spy_runner) -> None:
    runner = spy_runner()
    try:
        assert len(runner.task.attach_calls) == 1
        assert runner.task.attach_calls[0] == runner.strategy.device
    finally:
        runner.cleanup()


def test_step_end_called_per_optimizer_step(spy_runner) -> None:
    """accum_steps=1: 8 samples / batch 4 = 2 step_end calls."""
    runner = spy_runner()
    try:
        runner.run()
        assert runner.task.step_end_calls == 2
    finally:
        runner.cleanup()


def test_step_end_skipped_between_accumulation_batches(spy_runner) -> None:
    """accum_steps=2: 2 batches → 1 step_end."""
    runner = spy_runner(accum_steps=2)
    try:
        runner.run()
        assert runner.task.step_end_calls == 1
    finally:
        runner.cleanup()


def test_task_state_saved_to_checkpoint(spy_runner) -> None:
    runner = spy_runner()
    try:
        runner.run()
        ckpt_path = runner.artifacts.ckpt_path
    finally:
        runner.cleanup()

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "task" in ckpt
    assert ckpt["task"]["step_end_calls"] == 2
