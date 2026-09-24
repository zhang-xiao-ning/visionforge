"""End-to-end integration tests on real CIFAR-10 data.

These tests are skipped by default. To run them locally:

    RUN_INTEGRATION=1 uv run pytest tests/test_integration.py -v

They exercise the full stack: dataset loading, training, evaluation,
checkpoint saving, and resume. They are not run in CI because CI has
no CIFAR-10 data available.
"""

import os
from pathlib import Path

import pytest
import torch.optim as optim

from data.datasets import build_loaders
from experiment.config import TrainConfig
from experiment.runner import ExperimentRunner
from models.mlp import MLP
from tasks.classification import ClassificationTask
from training.strategy import SingleDeviceStrategy
from training.train import train
from utils.path import DATASETS_PATH

CIFAR10_PATH = DATASETS_PATH / "cifar-10-batches-py"
_HAS_CIFAR10 = (CIFAR10_PATH / "test_batch").exists()
_RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION", "").lower() in ("1", "true", "yes")

pytestmark = pytest.mark.skipif(
    not (_RUN_INTEGRATION and _HAS_CIFAR10),
    reason=(f"Integration tests require RUN_INTEGRATION=1 and CIFAR-10 under {CIFAR10_PATH}"),
)


def test_train_one_epoch_on_real_data() -> None:
    """Train 1 epoch on 128 real images. Verifies the full training path."""
    loader_train, loader_val, _ = build_loaders(name="cifar10", batch_size=32, num_train=128)

    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    result = train(model, optimizer, loader_train, loader_val, ClassificationTask(), epochs=1)

    assert result["last_epoch"] == 1
    assert 0.0 <= result["best_acc"] <= 1.0


def test_runner_saves_and_resumes(tmp_path: Path) -> None:
    """Run a small experiment, then resume from its checkpoint."""
    outputs_dir = tmp_path / "outputs"
    checkpoints_dir = tmp_path / "checkpoints"

    # First run: 1 epoch
    cfg1 = TrainConfig(experiment="mlp", epochs=1)
    runner1 = ExperimentRunner(
        cfg=cfg1,
        dataset_name="cifar10",
        batch_size=32,
        strategy=SingleDeviceStrategy(),
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
        num_train=128,
    )
    result1 = runner1.run()
    ckpt_path = runner1.artifacts.ckpt_path
    runner1.cleanup()

    assert ckpt_path.exists()
    assert result1["last_epoch"] == 1

    # Second run: resume from checkpoint, target epoch 2
    cfg2 = TrainConfig(experiment="mlp", epochs=2)
    runner2 = ExperimentRunner(
        cfg=cfg2,
        dataset_name="cifar10",
        batch_size=32,
        strategy=SingleDeviceStrategy(),
        resume_path=str(ckpt_path),
        outputs_dir=outputs_dir,
        checkpoints_dir=checkpoints_dir,
        num_train=128,
    )
    result2 = runner2.run()
    runner2.cleanup()

    assert result2["last_epoch"] == 2
    # Best accuracy after resume should be at least as good as after first run
    assert result2["best_acc"] >= result1["best_acc"]
