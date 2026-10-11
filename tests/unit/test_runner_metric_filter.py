"""Tests for metric filtering (run_every_n_epochs) in the runner."""

from pathlib import Path

import pytest

import application.registry  # noqa: F401
from application.registry import DATASETS, EXPERIMENTS
from fakes import TinyModel, TinyTask, make_classification_builder
from framework.experiment.runner import ExperimentRunner
from framework.experiment.spec import Experiment, TrainConfig
from framework.interfaces.metric import Metric
from framework.training import SingleDeviceStrategy


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


@pytest.fixture
def setup(monkeypatch, tmp_path: Path):
    def make(metrics: list[type[Metric]], primary: str) -> ExperimentRunner:
        fake_exp = Experiment(
            model=TinyModel,
            task=TinyTask,
            data="fake",
            metrics=metrics,
            primary_metric=primary,
            category="classification",
        )
        monkeypatch.setitem(EXPERIMENTS, "fake_exp", fake_exp)
        monkeypatch.setitem(
            DATASETS,
            "fake",
            make_classification_builder(shape=(4,), num_classes=2),
        )
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
