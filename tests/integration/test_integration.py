"""Integration tests: end-to-end training and resume."""

from pathlib import Path

import pytest

from experiment.config import TrainConfig
from experiment.runner import ExperimentRunner
from training.strategy import SingleDeviceStrategy

pytestmark = pytest.mark.integration


def test_train_one_epoch_on_real_data(require_integration, require_cifar10) -> None:
    cfg = TrainConfig(experiment="vit", epochs=1)
    runner = ExperimentRunner(
        cfg=cfg,
        batch_size=32,
        strategy=SingleDeviceStrategy(),
        num_train=32,
    )
    result = runner.run()
    runner.cleanup()

    assert result["last_epoch"] == 1
    assert 0.0 <= result["best_acc"] <= 1.0


def test_runner_saves_and_resumes(
    require_integration,
    require_cifar10,
    tmp_path: Path,
) -> None:
    outputs_dir = tmp_path / "outputs"
    checkpoints_dir = tmp_path / "checkpoints"

    cfg1 = TrainConfig(experiment="vit", epochs=1)
    runner1 = ExperimentRunner(
        cfg=cfg1,
        batch_size=32,
        strategy=SingleDeviceStrategy(),
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
        num_train=32,
    )
    result1 = runner1.run()
    ckpt_path = runner1.artifacts.ckpt_path
    runner1.cleanup()

    assert ckpt_path.exists()
    assert result1["last_epoch"] == 1

    cfg2 = TrainConfig(experiment="vit", epochs=2)
    runner2 = ExperimentRunner(
        cfg=cfg2,
        batch_size=32,
        strategy=SingleDeviceStrategy(),
        resume_path=str(ckpt_path),
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
        num_train=32,
    )
    result2 = runner2.run()
    runner2.cleanup()

    assert result2["last_epoch"] == 2
