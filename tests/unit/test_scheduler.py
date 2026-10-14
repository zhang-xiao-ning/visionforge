"""Tests for framework.experiment.scheduler.build_scheduler.

Verifies the design contract: warmup and main schedule compose freely,
all expressed as one step-based LambdaLR.
"""

import pytest
import torch

from framework.experiment.scheduler import build_scheduler
from framework.experiment.spec import TrainConfig

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detected call of `lr_scheduler.step\\(\\)` before:UserWarning"
)


def _optimizer() -> torch.optim.Optimizer:
    model = torch.nn.Linear(4, 2)
    return torch.optim.SGD(model.parameters(), lr=1.0)


def _lrs_after(scheduler, n_steps: int) -> list[float]:
    """lrs[0] = at construction; lrs[k] = after k .step() calls."""
    lrs = [scheduler.optimizer.param_groups[0]["lr"]]
    for _ in range(n_steps):
        scheduler.step()
        lrs.append(scheduler.optimizer.param_groups[0]["lr"])
    return lrs


def test_no_warmup_const_keeps_lr_flat() -> None:
    scheduler = build_scheduler(
        _optimizer(), TrainConfig(epochs=1, lr_scheduler="const"), steps_per_epoch=10
    )
    assert all(abs(lr - 1.0) < 1e-9 for lr in _lrs_after(scheduler, 5))


def test_warmup_ramps_linearly_then_const() -> None:
    cfg = TrainConfig(epochs=1, warmup_steps=4, lr_scheduler="const")
    scheduler = build_scheduler(_optimizer(), cfg, steps_per_epoch=10)
    lrs = _lrs_after(scheduler, 5)
    assert lrs[0] == 0.0  # construction
    assert abs(lrs[1] - 0.25) < 1e-9
    assert abs(lrs[2] - 0.5) < 1e-9
    assert abs(lrs[3] - 0.75) < 1e-9
    assert abs(lrs[4] - 1.0) < 1e-9  # warmup done, const -> 1.0
    assert abs(lrs[5] - 1.0) < 1e-9


def test_warmup_plus_step_decay_is_honored() -> None:
    """Regression: warmup_steps > 0 must NOT silently ignore 'step'."""
    cfg = TrainConfig(
        epochs=2,
        warmup_steps=2,
        lr_scheduler="step",
        step_size=1,
        gamma=0.5,
    )
    scheduler = build_scheduler(_optimizer(), cfg, steps_per_epoch=4)
    # opt_steps_per_epoch = 4, so every = 1 * 4 = 4 optimizer steps.
    lrs = _lrs_after(scheduler, 10)
    assert lrs[0] == 0.0  # warmup start
    assert abs(lrs[1] - 0.5) < 1e-9  # warmup
    assert abs(lrs[2] - 1.0) < 1e-9  # s=0 -> 0 decay
    assert abs(lrs[6] - 0.5) < 1e-9  # s=4 -> 1 decay
    assert abs(lrs[10] - 0.25) < 1e-9  # s=8 -> 2 decays


def test_warmup_plus_cosine_descends_to_zero() -> None:
    cfg = TrainConfig(epochs=1, warmup_steps=2, lr_scheduler="cosine")
    scheduler = build_scheduler(_optimizer(), cfg, steps_per_epoch=10)
    # total_steps = 10, warmup = 2, remaining = 8.
    lrs = _lrs_after(scheduler, 10)
    assert lrs[0] == 0.0
    assert abs(lrs[2] - 1.0) < 1e-9  # warmup ends, cos(0) = 1
    assert abs(lrs[10]) < 1e-9  # cos(pi) = -1 -> 0


def test_cosine_without_warmup() -> None:
    cfg = TrainConfig(epochs=1, lr_scheduler="cosine")
    scheduler = build_scheduler(_optimizer(), cfg, steps_per_epoch=10)
    lrs = _lrs_after(scheduler, 10)
    assert abs(lrs[0] - 1.0) < 1e-9
    assert abs(lrs[5] - 0.5) < 1e-9
    assert abs(lrs[10]) < 1e-9


def test_unknown_scheduler_raises_at_construction() -> None:
    """LambdaLR calls lr_lambda(0) at construction, so the error fires
    before _build_scheduler returns."""
    with pytest.raises(ValueError, match="Unknown lr_scheduler"):
        build_scheduler(_optimizer(), TrainConfig(lr_scheduler="nope"), steps_per_epoch=10)
