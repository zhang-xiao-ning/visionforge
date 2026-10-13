"""Tests for per-group learning rates (TrainConfig.lr_per_group)."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from fakes import make_classification_builder
from framework.experiment.runner import ExperimentRunner
from framework.experiment.spec import Experiment, TrainConfig
from framework.interfaces.model import Model
from framework.interfaces.task import Task
from framework.metrics import Accuracy
from framework.registry import DATASETS, EXPERIMENTS
from framework.training import SingleDeviceStrategy


class _TwoGroupModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {"a": list(self.a.parameters()), "b": list(self.b.parameters())}


class _ClassifyTask(Task):
    def train_step(self, model, batch, device, dtype):
        x, y = batch
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        return nn.functional.cross_entropy(model(x), y)


@pytest.fixture
def make_runner(monkeypatch, tmp_path: Path):
    exp = Experiment(
        name="lr_exp",
        model=_TwoGroupModel,
        task=_ClassifyTask,
        data="fake",
        metrics=[Accuracy],
        primary_metric="acc",
        category="classification",
    )
    monkeypatch.setitem(EXPERIMENTS, "lr_exp", exp)
    monkeypatch.setitem(
        DATASETS, "fake", make_classification_builder(shape=(4,), num_classes=2, model_init={})
    )

    def _make(lr_per_group: dict[str, float]) -> ExperimentRunner:
        return ExperimentRunner(
            experiment=EXPERIMENTS["lr_exp"],
            config=TrainConfig(lr_per_group=lr_per_group),
            strategy=SingleDeviceStrategy(),
            outputs_dir=tmp_path / "outputs",
            checkpoints_dir=tmp_path / "checkpoints",
        )

    return _make


def test_default_lr_applies_to_every_group(make_runner) -> None:
    runner = make_runner({})
    try:
        lrs = [g["lr"] for g in runner.optimizer.param_groups]
        assert lrs == [runner.cfg.learning_rate] * 2
    finally:
        runner.cleanup()


def test_lr_per_group_overrides(make_runner) -> None:
    runner = make_runner({"a": 1e-3, "b": 1e-4})
    try:
        lrs = [g["lr"] for g in runner.optimizer.param_groups]
        assert lrs == [1e-3, 1e-4]
    finally:
        runner.cleanup()


def test_partial_lr_per_group_falls_back_to_default(make_runner) -> None:
    runner = make_runner({"a": 1e-3})
    try:
        lrs = [g["lr"] for g in runner.optimizer.param_groups]
        assert lrs == [1e-3, runner.cfg.learning_rate]
    finally:
        runner.cleanup()


def test_unknown_group_raises(make_runner) -> None:
    with pytest.raises(ValueError, match="Unknown lr_per_group keys"):
        make_runner({"nonexistent": 1e-3})
