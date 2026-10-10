"""Hooks for the training loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn as nn
from torch.utils.data import DataLoader

from framework.utils import AppLogger


@dataclass
class StepContext:
    epoch: int
    step: int
    loss: float


@dataclass
class EpochContext:
    """Everything hooks need at epoch end.

    Only holds resources that `train()` itself owns: the model and the
    validation loader. Metrics live on the runner and are accessed via
    the hook closure.
    """

    epoch: int
    avg_loss: float
    lr: float
    model: nn.Module
    loader_val: DataLoader
    val_metrics: dict[str, float] = field(default_factory=dict)


def _noop_step(ctx: StepContext) -> None:
    return


def _noop_epoch_end(ctx: EpochContext) -> bool:
    return True


def _noop_train_end() -> None:
    return


def _empty_result() -> dict[str, float]:
    return {}


@dataclass
class TrainHooks:
    logger: AppLogger | None = None
    on_step: Callable[[StepContext], None] = _noop_step
    on_epoch_end: Callable[[EpochContext], bool] = _noop_epoch_end
    on_train_end: Callable[[], None] = _noop_train_end
    result: Callable[[], dict[str, float]] = _empty_result
