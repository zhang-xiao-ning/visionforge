"""Regression tests: verify metrics do not collapse after refactors.

Each entry is (experiment, epochs, num_train, direction, threshold).
`direction` is "acc>=" for higher-is-better metrics and "ppl<=" for
lower-is-better. Thresholds are intentionally loose: they catch
"training broke" (weights not updating, loss not descending, EMA
never applied), not "the last 2% of accuracy is still there".
"""

import dataclasses

import pytest

from framework.experiment.runner import ExperimentRunner
from framework.registry import EXPERIMENTS
from framework.training import SingleDeviceStrategy

pytestmark = pytest.mark.regression


BASELINES = [
    ("mlp", 1, 1000, "acc>=", 0.15),
    ("nanogpt", 1, None, "ppl<=", 25.0),
]


@pytest.mark.parametrize(
    "experiment,epochs,num_train,direction,threshold",
    BASELINES,
    ids=[f"{e}-{d}-{t}" for e, _, _, d, t in BASELINES],
)
def test_metric_within_baseline(
    require_regression,
    require_cifar10,
    require_tinyshakespeare,
    experiment: str,
    epochs: int,
    num_train: int | None,
    direction: str,
    threshold: float,
    tmp_path,
) -> None:
    exp = EXPERIMENTS[experiment]
    cfg = dataclasses.replace(exp.config, epochs=epochs)
    runner = ExperimentRunner(
        experiment=exp,
        config=cfg,
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
        num_train=num_train,
    )
    result = runner.run()
    runner.cleanup()

    # Primary metric value comes from `best_acc` in the result dict —
    # the field name is historical; it holds the primary metric's best.
    value = result["best_acc"]
    metric = exp.primary_metric

    if direction == "acc>=":
        assert value >= threshold, (
            f"{experiment}: {metric}={value:.4f} below baseline {threshold}. "
            f"Possible regression in training logic."
        )
    elif direction == "ppl<=":
        assert value <= threshold, (
            f"{experiment}: {metric}={value:.4f} above baseline {threshold}. "
            f"Possible regression in training logic."
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown direction: {direction}")
