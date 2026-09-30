"""Hooks for the training loop.

The training loop calls these at well-defined points. Their default
implementations are no-ops, so worker processes (in DDP) can use the
defaults. The master process builds real hooks in ExperimentRunner.

Why a dataclass of callables instead of an ABC:
- Python's first-class functions make "master vs worker" a difference
  of *which functions are passed*, not which class inherits from which.
- No inheritance, no boilerplate, one place to see the divergence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader

from utils.logger import AppLogger


@dataclass
class StepContext:
    """Everything hooks need at one training step."""

    epoch: int
    step: int
    loss: float


@dataclass
class EpochContext:
    """Everything hooks need at epoch end."""

    epoch: int
    avg_loss: float
    lr: float
    model: Any  # nn.Module
    loader_val: DataLoader
    task: Any  # Task
    val_metrics: dict[str, float] = field(default_factory=dict)


def _noop_step(ctx: StepContext) -> None:
    return


def _noop_epoch_end(ctx: EpochContext) -> bool:
    """Return True to continue, False to stop. Default: continue."""
    return True


def _noop_train_end() -> None:
    return


def _empty_result() -> dict[str, float]:
    return {}


@dataclass
class TrainHooks:
    """Collection of training-time hooks.

    All default to no-ops (worker-process behavior). The master process
    passes real callables.
    """

    logger: AppLogger | None = None
    on_step: Callable[[StepContext], None] = _noop_step
    on_epoch_end: Callable[[EpochContext], bool] = _noop_epoch_end
    on_train_end: Callable[[], None] = _noop_train_end
    result: Callable[[], dict[str, float]] = _empty_result
