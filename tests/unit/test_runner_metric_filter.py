"""Tests for metric filtering (run_every_n_epochs) in the runner."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from application.registry import EXPERIMENTS
from framework.experiment.runner import ExperimentRunner
from framework.experiment.spec import Experiment, TrainConfig
from framework.interfaces.bundle import DataBundle
from framework.interfaces.metric import Metric
from framework.interfaces.model import Model
from framework.interfaces.task import Task
from framework.training import SingleDeviceStrategy


class _TinyModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _TinyTask(Task):
    def train_step(self, model, batch, device, dtype):
        x, y = batch
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        return nn.functional.cross_entropy(model(x), y)


class _CountingBase(Metric):
    """Counts evaluate() calls in a class-level counter."""

    higher_is_better = True
    run_every_n_epochs: int | None = 1
    calls: int = 0

    def evaluate(self, model, loader, device, dtype) -> float:
        type(self).calls += 1
        return 0.5


class _Primary(_CountingBase):
    name = "primary"


class _Slow(_CountingBase):
    name = "slow"
    run_every_n_epochs = None


class _Every2(_CountingBase):
    name = "every2"
    run_every_n_epochs = 2


@pytest.fixture(autouse=True)
def _reset_counters():
    _Primary.calls = 0
    _Slow.calls = 0
    _Every2.calls = 0
    yield


def _fake_data_bundle(name: str, ctx) -> tuple[DataBundle, None]:  # noqa: ARG001
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    return DataBundle(
        loader_train=loader,
        loader_val=loader,
        loader_test=loader,
        model_init={},
        extras={},
    ), None


@pytest.fixture
def setup(monkeypatch, tmp_path: Path):
    def make(metrics: list[type[Metric]], primary: str) -> ExperimentRunner:
        fake_exp = Experiment(
            model=_TinyModel,
            task=_TinyTask,
            data="fake",
            metrics=metrics,
            primary_metric=primary,
            category="classification",
        )
        monkeypatch.setitem(EXPERIMENTS, "fake_exp", fake_exp)
        monkeypatch.setattr("framework.experiment.runner.build_data", _fake_data_bundle)
        return ExperimentRunner(
            experiment=EXPERIMENTS.get("fake_exp"),
            experiment_name="fake_exp",
            config=TrainConfig(epochs=3),
            strategy=SingleDeviceStrategy(),
            outputs_dir=tmp_path / "outputs",
            checkpoints_dir=tmp_path / "checkpoints",
        )

    return make


def test_none_metric_runs_only_at_test(setup) -> None:
    runner = setup([_Primary, _Slow], primary="primary")
    runner.run()
    runner.cleanup()

    # primary: 3 train epochs + 1 test = 4
    assert _Primary.calls == 4
    # slow: 0 train + 1 test = 1
    assert _Slow.calls == 1


def test_every_2_epochs(setup) -> None:
    runner = setup([_Primary, _Every2], primary="primary")
    runner.run()
    runner.cleanup()

    # 3 epochs: primary runs every epoch (3) + test (1) = 4
    assert _Primary.calls == 4
    # every2 runs on epoch 2 only + test = 2
    assert _Every2.calls == 2


def test_primary_metric_must_run_every_epoch(setup) -> None:
    with pytest.raises(ValueError, match="must run every epoch"):
        setup([_Every2], primary="every2")


def test_primary_metric_cannot_be_test_only(setup) -> None:
    with pytest.raises(ValueError, match="must run every epoch"):
        setup([_Slow], primary="slow")
