"""Integration tests for ExperimentRunner end-to-end."""

import pytest
from torch.utils.data import SubsetRandomSampler

from data.datasets import build_data
from experiment.runner import ExperimentRunner
from experiment.spec import TrainConfig
from framework.interfaces.bundle import DataContext
from training.strategy import SingleDeviceStrategy

pytestmark = pytest.mark.integration


def test_runner_runs_one_epoch_on_real_data(
    require_integration,
    require_cifar10,
    tmp_path,
) -> None:
    """Full run: build data → train 1 epoch → save checkpoint."""
    runner = ExperimentRunner(
        experiment_name="mlp",
        config=TrainConfig(epochs=1, batch_size=64),
        strategy=SingleDeviceStrategy(),
        num_train=32,
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    try:
        result = runner.run()
        ckpt_path = runner.artifacts.ckpt_path
    finally:
        runner.cleanup()

    assert result["last_epoch"] == 1
    assert "test_acc" in result
    assert ckpt_path.exists()


def test_runner_saves_and_resumes(
    require_integration,
    require_cifar10,
    tmp_path,
) -> None:
    """Train 1 epoch → checkpoint → resume → train 1 more epoch."""
    outputs_dir = tmp_path / "outputs"
    checkpoints_dir = tmp_path / "checkpoints"

    runner = ExperimentRunner(
        experiment_name="mlp",
        config=TrainConfig(epochs=1, batch_size=64),
        strategy=SingleDeviceStrategy(),
        num_train=32,
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
    )
    try:
        runner.run()
        ckpt = runner.artifacts.ckpt_path
    finally:
        runner.cleanup()
    assert ckpt.exists()

    runner2 = ExperimentRunner(
        experiment_name="mlp",
        config=TrainConfig(epochs=2, batch_size=64),
        strategy=SingleDeviceStrategy(),
        num_train=32,
        resume_path=str(ckpt),
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
    )
    try:
        result2 = runner2.run()
    finally:
        runner2.cleanup()

    assert result2["last_epoch"] == 2


def test_flickr8k_train_loader_uses_strategy_sampler(require_flickr8k) -> None:
    """Regression: train loader must route through strategy.make_train_sampler."""
    from training.strategy import SingleDeviceStrategy

    bundle, _ = build_data(
        "flickr8k",
        DataContext(batch_size=4, strategy=SingleDeviceStrategy(), num_train=4),
    )
    assert isinstance(bundle.loader_train.sampler, SubsetRandomSampler)
