"""Metric base class.

A Metric knows how to score a model on a whole loader. Unlike `Task`
(which only defines the training loss), a Metric drives its own
evaluation loop: it decides what to extract from each batch, how to
run the model, and how to aggregate.

Stateless contract: `evaluate()` must be safe to call repeatedly.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from data.bundle import DataBundle


class Metric(ABC):
    #: Name used as the key in metrics dicts / CSV columns.
    name: str

    #: Whether larger values are better.
    higher_is_better: bool

    #: During training, run every N epochs.
    #: None = skip during training (only run at final test evaluation).
    run_every_n_epochs: int | None

    @classmethod
    def from_data(cls, bundle: "DataBundle") -> Self:
        """Build the metric from a DataBundle. Default: cls()."""
        return cls()

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        """Score `model` over `loader`. Returns a single float."""
