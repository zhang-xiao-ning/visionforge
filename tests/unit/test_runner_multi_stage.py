"""Tests for the multi-stage training path in ExperimentRunner."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.bundle import DataBundle
from experiment.config import TrainConfig
from experiment.runner import ExperimentRunner
from models.base import Model
from registry import EXPERIMENTS
from tasks.base import Stage, Task
from training.strategy import SingleDeviceStrategy


class _TwoGroupModel(Model):
    """Minimal model exposing two named param groups: 'a' and 'b'."""

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
    """Classification-ish task with a two-stage plan."""

    primary_metric = "loss"
    higher_is_better = False
    stages = [
        Stage(name="phase1", epochs=1, freeze=["a"]),
        Stage(name="phase2", epochs=2, freeze=[]),
    ]

    @classmethod
    def from_data(cls, bundle: DataBundle) -> "_MultiStageTask":
        return cls()

    def train_step(self, model, batch, device, dtype):
        x, y = batch
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        return nn.functional.cross_entropy(model(x), y)

    def eval_step(self, model, batch, device, dtype):
        x, y = batch
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        loss = nn.functional.cross_entropy(model(x), y)
        return {"loss": loss.item()}


def _fake_data_bundle(name: str, ctx) -> DataBundle:  # noqa: ARG001
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    return DataBundle(
        loader_train=loader,
        loader_val=loader,
        loader_test=loader,
        model_init={},
        extras={},
    )


@pytest.fixture
def patched_runner(monkeypatch, tmp_path: Path):
    """Build an ExperimentRunner with fake data/model/task injected."""

    fake_exp = {
        "data": "fake",
        "model": _TwoGroupModel,
        "task": _MultiStageTask,
        "lr": 1e-3,
        "category": "classification",
    }
    monkeypatch.setitem(EXPERIMENTS, "fake_exp", fake_exp)
    monkeypatch.setattr("experiment.runner.build_data", _fake_data_bundle)

    def make() -> ExperimentRunner:
        cfg = TrainConfig(experiment="fake_exp", epochs=999)  # 999 ignored
        return ExperimentRunner(
            cfg=cfg,
            batch_size=4,
            strategy=SingleDeviceStrategy(),
            outputs_dir=tmp_path / "outputs",
            checkpoints_dir=tmp_path / "checkpoints",
        )

    return make


def test_multi_stage_runs_all_stages(patched_runner) -> None:
    runner = patched_runner()
    result = runner.run()
    runner.cleanup()
    # Last stage had 2 epochs → last_epoch == 2
    assert result["last_epoch"] == 2


def test_multi_stage_ignores_cfg_epochs(patched_runner) -> None:
    """cfg.epochs=999 must not leak into the multi-stage run."""
    runner = patched_runner()
    result = runner.run()
    runner.cleanup()
    # last stage's epoch count, not cfg.epochs
    assert result["last_epoch"] == 2


def test_multi_stage_resume_raises(patched_runner, tmp_path: Path) -> None:
    """Resume is not supported for multi-stage training."""
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
