"""Task interface."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from data.bundle import DataBundle


class Task(ABC):
    """Encapsulates task-specific logic (loss, metrics).

    A Task does NOT know about:
    - optimizers / schedulers
    - AMP / DDP
    - logging / checkpointing
    - training stages

    It only knows: given a model and a batch, how to compute loss / metrics.
    """

    # Which metric from `eval_step` should be used for best-model selection
    primary_metric: str = "loss"

    # Whether a larger primary_metric is better (True for accuracy,
    # False for loss / perplexity)
    higher_is_better: bool = True

    @classmethod
    def from_data(cls, bundle: "DataBundle") -> Self:
        """Build the task from a DataBundle. Default: cls()."""
        return cls()

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
