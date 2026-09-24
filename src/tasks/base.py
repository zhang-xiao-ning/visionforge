"""Task interface."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class Task(ABC):
    """Encapsulates task-specific logic (loss, metrics).

    A Task does NOT know about:
    - optimizers / schedulers
    - AMP / DDP
    - logging / checkpointing

    It only knows: given a model and a batch, how to compute loss / metrics.
    """

    # Which metric from `eval_step` should be used for best-model selection
    primary_metric: str = "loss"

    @abstractmethod
    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute the loss for one batch. Must NOT call backward."""

    @abstractmethod
    def eval_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, float]:
        """Compute metrics for one batch. Returns a dict of scalar floats."""
