"""Smoke tests for ExperimentRunner."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiment.config import TrainConfig
from experiment.runner import ExperimentRunner
from training.strategy import SingleDeviceStrategy


def _fake_build_loaders(name: str, batch_size: int, **kwargs):
    """Return tiny loaders with the right shape for MLP."""
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(x, y)
    return (
        DataLoader(dataset, batch_size=4),
        DataLoader(dataset, batch_size=4),
        DataLoader(dataset, batch_size=4),
    )


def test_runner_runs_one_epoch(monkeypatch, tmp_path: Path, any_experiment: str) -> None:
    monkeypatch.setattr("experiment.runner.build_loaders", _fake_build_loaders)

    cfg = TrainConfig(experiment=any_experiment, epochs=1)
    runner = ExperimentRunner(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=4,
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


def test_runner_saves_checkpoint(monkeypatch, tmp_path: Path, any_experiment: str) -> None:
    monkeypatch.setattr("experiment.runner.build_loaders", _fake_build_loaders)

    cfg = TrainConfig(experiment=any_experiment, epochs=1)
    runner = ExperimentRunner(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=4,
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    runner.run()
    runner.cleanup()

    assert runner.artifacts.ckpt_path.exists()


def test_runner_passes_num_train_to_build_loaders(
    monkeypatch, tmp_path: Path, any_experiment: str
) -> None:
    captured: dict[str, int] = {}

    def fake_build_loaders(name: str, batch_size: int, num_train: int, **kwargs):
        captured["num_train"] = num_train
        return _fake_build_loaders(name, batch_size)

    monkeypatch.setattr("experiment.runner.build_loaders", fake_build_loaders)

    cfg = TrainConfig(experiment=any_experiment, epochs=1)
    runner = ExperimentRunner(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=4,
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
        num_train=128,
    )
    runner.cleanup()

    assert captured["num_train"] == 128
