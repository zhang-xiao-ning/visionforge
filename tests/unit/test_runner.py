"""Smoke tests for ExperimentRunner."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from data.bundle import DataBundle
from experiment.runner import ExperimentRunner
from experiment.spec import TrainConfig
from training.strategy import SingleDeviceStrategy


def _fake_data_bundle(name: str, ctx) -> DataBundle:  # noqa: ARG001
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=4)
    return DataBundle(
        loader_train=loader,
        loader_val=loader,
        loader_test=loader,
        model_init={"num_classes": 10},
        extras={},
    )


def test_runner_runs_one_epoch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("experiment.runner.build_data", _fake_data_bundle)

    runner = ExperimentRunner(
        experiment_name="vit",
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
    monkeypatch.setattr("experiment.runner.build_data", _fake_data_bundle)

    runner = ExperimentRunner(
        experiment_name="vit",
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

    def fake_build_data(name: str, ctx):
        captured["num_train"] = ctx.num_train
        return _fake_data_bundle(name, ctx)

    monkeypatch.setattr("experiment.runner.build_data", fake_build_data)

    runner = ExperimentRunner(
        experiment_name="vit",
        config=TrainConfig(),
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
        num_train=128,
    )
    runner.cleanup()

    assert captured["num_train"] == 128
