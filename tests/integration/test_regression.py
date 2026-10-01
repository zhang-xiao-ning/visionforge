"""Regression tests: verify model accuracy does not collapse after refactors."""

import dataclasses

import pytest

from experiment.runner import ExperimentRunner
from registry import EXPERIMENTS
from training.strategy import SingleDeviceStrategy

pytestmark = pytest.mark.regression


BASELINES = [
    ("mlp", 1, 1000, 0.15),
]


@pytest.mark.parametrize(
    "experiment,epochs,num_train,min_val_acc",
    BASELINES,
    ids=[f"{e}-{ep}ep-{n}imgs" for e, ep, n, _ in BASELINES],
)
def test_accuracy_above_baseline(
    require_regression,
    require_cifar10,
    experiment: str,
    epochs: int,
    num_train: int,
    min_val_acc: float,
    tmp_path,
) -> None:
    exp = EXPERIMENTS[experiment]
    cfg = dataclasses.replace(exp.config, epochs=epochs)
    runner = ExperimentRunner(
        experiment_name=experiment,
        config=cfg,
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
