"""Tests for the multi-stage training path in ExperimentRunner."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.bundle import DataBundle
from evaluation import CrossEntropy
from experiment.runner import ExperimentRunner
from experiment.spec import Experiment, Stage, TrainConfig
from models.base import Model
from registry import EXPERIMENTS
from tasks.base import Task
from training.strategy import SingleDeviceStrategy


class _TwoGroupModel(Model):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))

    def param_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "a": list(self.a.parameters()),
            "b": list(self.b.parameters()),
        }


class _MultiStageTask(Task):
    primary_metric = "loss"
    higher_is_better = False

    def train_step(self, model, batch, device, dtype):
        x, y = batch
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        return nn.functional.cross_entropy(model(x), y)


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


_DEFAULT_STAGES = [
    Stage(name="phase1", freeze=["a"], overrides={"epochs": 1}),
    Stage(name="phase2", freeze=[], overrides={"epochs": 2}),
]


@pytest.fixture
def patched_runner(monkeypatch, tmp_path: Path):
    fake_exp = Experiment(
        model=_TwoGroupModel,
        task=_MultiStageTask,
        data="fake",
        metrics=[CrossEntropy],
        primary_metric="loss",
        category="classification",
    )
    monkeypatch.setitem(EXPERIMENTS, "fake_exp", fake_exp)
    monkeypatch.setattr("experiment.runner.build_data", _fake_data_bundle)

    def make(stages=None) -> ExperimentRunner:
        return ExperimentRunner(
            experiment_name="fake_exp",
            config=TrainConfig(),
            stages=stages if stages is not None else _DEFAULT_STAGES,
            strategy=SingleDeviceStrategy(),
            outputs_dir=tmp_path / "outputs",
            checkpoints_dir=tmp_path / "checkpoints",
        )

    return make


def test_multi_stage_runs_all_stages(patched_runner) -> None:
    runner = patched_runner()
    result = runner.run()
    runner.cleanup()
    assert result["last_epoch"] == 2


def test_multi_stage_respects_stage_epochs(patched_runner) -> None:
    """Last stage has 2 epochs → last_epoch == 2, regardless of TrainConfig.epochs."""
    runner = patched_runner()
    result = runner.run()
    runner.cleanup()
    assert result["last_epoch"] == 2


def test_multi_stage_resume_raises(patched_runner, tmp_path: Path) -> None:
    runner = patched_runner()
    runner.resume_path = str(tmp_path / "fake.pt")
    with pytest.raises(NotImplementedError, match="Resume is not supported"):
        runner.run()
    runner.cleanup()


def test_apply_freeze_unknown_group_raises(patched_runner) -> None:
    runner = patched_runner()
    with pytest.raises(ValueError, match="Unknown freeze groups"):
        runner._apply_freeze(["nope"])
    runner.cleanup()


def test_apply_freeze_sets_requires_grad(patched_runner) -> None:
    runner = patched_runner()
    runner._apply_freeze(["a"])
    for p in runner.model.a.parameters():
        assert p.requires_grad is False
    for p in runner.model.b.parameters():
        assert p.requires_grad is True
    runner.cleanup()


def test_apply_freeze_unfreeze_all(patched_runner) -> None:
    runner = patched_runner()
    runner._apply_freeze(["a"])
    runner._apply_freeze([])
    for p in runner.model.a.parameters():
        assert p.requires_grad is True
    runner.cleanup()
