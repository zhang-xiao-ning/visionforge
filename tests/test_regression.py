"""Regression tests: verify model accuracy does not collapse after refactors.

These tests are skipped by default. To run them locally:

    RUN_REGRESSION=1 uv run pytest tests/test_regression.py -v

They train on a small subset of CIFAR-10 (1000 images, 1 epoch) and assert
that validation accuracy stays above a loose baseline. The thresholds are
deliberately generous: they catch "training is broken" bugs, not minor
numerical drift from environment differences (MPS vs CUDA vs CPU).
"""

import os

import pytest

from experiment.config import TrainConfig
from experiment.runner import ExperimentRunner
from registry import EXPERIMENTS
from training.strategy import SingleDeviceStrategy
from utils.path import DATASETS_PATH

CIFAR10_PATH = DATASETS_PATH / "cifar-10-batches-py"
_HAS_CIFAR10 = (CIFAR10_PATH / "test_batch").exists()
_RUN_REGRESSION = os.environ.get("RUN_REGRESSION", "").lower() in ("1", "true", "yes")

pytestmark = pytest.mark.skipif(
    not (_RUN_REGRESSION and _HAS_CIFAR10),
    reason=(f"Regression tests require RUN_REGRESSION=1 and CIFAR-10 under {CIFAR10_PATH}"),
)


# (experiment, epochs, num_train, min_val_acc)
BASELINES = [
    ("mlp", 1, 1000, 0.15),
]


@pytest.mark.parametrize(
    "experiment,epochs,num_train,min_val_acc",
    BASELINES,
    ids=[f"{e}-{ep}ep-{n}imgs" for e, ep, n, _ in BASELINES],
)
def test_accuracy_above_baseline(
    experiment: str,
    epochs: int,
    num_train: int,
    min_val_acc: float,
    tmp_path,
) -> None:
    _, default_lr = EXPERIMENTS[experiment]
    cfg = TrainConfig(experiment=experiment, epochs=epochs, learning_rate=default_lr)
    runner = ExperimentRunner(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=64,
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
        num_train=num_train,
    )
    result = runner.run()
    runner.cleanup()

    actual = result["best_acc"]
    assert actual >= min_val_acc, (
        f"{experiment}: val_acc={actual:.4f} below baseline {min_val_acc}. "
        f"Possible regression in training logic."
    )
