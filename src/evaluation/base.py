"""Metric Protocol.

A Metric knows how to score a model on a whole loader. Unlike `Task`
(which only defines the training loss), a Metric drives its own
evaluation loop: it decides what to extract from each batch, how to
run the model, and how to aggregate.

Stateless contract: `evaluate()` must be safe to call repeatedly. Any
accumulator lives in local variables inside `evaluate()`.
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@runtime_checkable
class Metric(Protocol):
    #: Name used as the key in metrics dicts / CSV columns.
    name: str

    #: Whether larger values are better (True for accuracy, False for loss).
    higher_is_better: bool

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> float:
        """Score `model` over `loader`. Returns a single float."""
