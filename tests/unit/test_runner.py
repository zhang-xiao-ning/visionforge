"""Smoke tests for ExperimentRunner."""

from pathlib import Path

from framework.experiment.runner import ExperimentRunner
from framework.experiment.spec import TrainConfig
from framework.registry import DATASETS, EXPERIMENTS
from framework.training import SingleDeviceStrategy
from tests.fakes import make_classification_builder


def test_runner_runs_one_epoch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(DATASETS, "cifar10", make_classification_builder())

    runner = ExperimentRunner(
        experiment=EXPERIMENTS["vit"],
        config=TrainConfig(),
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    result = runner.run()
    runner.cleanup()

    assert "test_acc" in result
    assert "best_acc" in result
    assert "last_epoch" in result
    assert result["last_epoch"] == 1


def test_runner_saves_checkpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setitem(DATASETS, "cifar10", make_classification_builder())
    runner = ExperimentRunner(
        experiment=EXPERIMENTS["vit"],
        config=TrainConfig(),
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    runner.run()
    runner.cleanup()

    assert runner.artifacts.ckpt_path.exists()


def test_runner_passes_num_train_to_build_data(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, int | None] = {}

    base = make_classification_builder()

    def recording_builder(ctx):
        captured["num_train"] = ctx.num_train
        return base(ctx)

    monkeypatch.setitem(DATASETS, "cifar10", recording_builder)

    runner = ExperimentRunner(
        experiment=EXPERIMENTS["vit"],
        config=TrainConfig(),
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
        num_train=128,
    )
    runner.cleanup()

    assert captured["num_train"] == 128
