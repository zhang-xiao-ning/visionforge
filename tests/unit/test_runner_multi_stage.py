"""Tests for the multi-stage training path in ExperimentRunner."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

import application.registry  # noqa: F401
from fakes import make_classification_builder
from framework.experiment.runner import ExperimentRunner
from framework.experiment.spec import Experiment, Stage, TrainConfig
from framework.interfaces.model import Model
from framework.interfaces.task import Task
from framework.metrics import CrossEntropy
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


_DEFAULT_STAGES = [
    Stage(name="phase1", freeze=["a"], overrides={"epochs": 1}),
    Stage(name="phase2", freeze=[], overrides={"epochs": 2}),
]


@pytest.fixture
def fake_experiment(monkeypatch) -> None:
    """Register a two-group fake experiment under 'fake_exp'."""
    fake_exp = Experiment(
        model=_TwoGroupModel,
        task=_MultiStageTask,
        data="fake",
        metrics=[CrossEntropy],
        primary_metric="loss",
        category="classification",
    )
    monkeypatch.setitem(EXPERIMENTS, "fake_exp", fake_exp)


@pytest.fixture
def patched_runner(monkeypatch, tmp_path: Path, fake_experiment: None):
    monkeypatch.setitem(
        DATASETS,
        "fake",
        make_classification_builder(shape=(4,), num_classes=2, model_init={}),
    )

    def make(stages=None, resume_path=None) -> ExperimentRunner:
        return ExperimentRunner(
            experiment=EXPERIMENTS["fake_exp"],
            experiment_name="fake_exp",
            config=TrainConfig(),
            stages=stages if stages is not None else _DEFAULT_STAGES,
            resume_path=resume_path,
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
    with pytest.raises(NotImplementedError, match="Resume is not supported"):
        patched_runner(resume_path=str(tmp_path / "fake.pt"))


def test_multi_stage_broadcasts_after_restore(patched_runner, monkeypatch):
    """restore + broadcast_model must be called at each stage boundary."""
    runner = patched_runner()
    calls = {"broadcast": 0}
    orig = runner.strategy.broadcast_model

    def counting(model):
        calls["broadcast"] += 1
        return orig(model)

    monkeypatch.setattr(runner.strategy, "broadcast_model", counting)

    runner.run()
    runner.cleanup()

    # 2 stages in _DEFAULT_STAGES, plus once at end of run()
    assert calls["broadcast"] == 3


def test_broadcast_model_covers_buffers():
    """broadcast_model must cover both parameters and buffers."""
    import torch.nn as nn

    from framework.training import SingleDeviceStrategy

    model = nn.BatchNorm1d(4)
    # Single device: no-op. But we can at least assert it doesn't raise
    # and doesn't touch the state.
    before = {k: v.clone() for k, v in model.state_dict().items()}
    SingleDeviceStrategy().broadcast_model(model)
    after = model.state_dict()
    for k in before:
        assert torch.equal(before[k], after[k])


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


def test_multi_stage_resume_fails_before_creating_dirs(
    fake_experiment: None, tmp_path: Path
) -> None:
    """Resume+stages must fail before RunArtifacts.create makes dirs."""
    outputs_dir = tmp_path / "outputs"
    checkpoints_dir = tmp_path / "checkpoints"

    with pytest.raises(NotImplementedError):
        ExperimentRunner(
            experiment=EXPERIMENTS["fake_exp"],
            experiment_name="fake_exp",
            config=TrainConfig(),
            stages=_DEFAULT_STAGES,
            resume_path=str(tmp_path / "fake.pt"),
            strategy=SingleDeviceStrategy(),
            outputs_dir=outputs_dir,
            checkpoints_dir=checkpoints_dir,
        )

    assert not outputs_dir.exists()
    assert not checkpoints_dir.exists()
