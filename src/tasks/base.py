"""Task interface.

A Task defines *what to optimize*: given a model and a batch, how to
compute the training loss. It does not define metrics — those are
first-class objects (`src/evaluation/`), declared per experiment.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from data.bundle import DataBundle


class Task(ABC):
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
