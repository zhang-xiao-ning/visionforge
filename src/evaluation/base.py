"""Metric base class.

A Metric knows how to score a model. Unlike `Task` (which defines the
training loss), a Metric drives its own evaluation loop: it decides
what loader to use, what to extract from each batch, and how to
aggregate.

Two hooks let a Metric declare its data source:
- `train_loader`: used during training (each N epochs)
- `test_loader`: used after training

Both default to the DataBundle's val / test loaders. A Metric that
needs different data (e.g. BLEU needs image-level batches with all
references) overrides them.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from data.bundle import DataBundle, EvalBundle


class Metric(ABC):
    #: Name used as the key in metrics dicts / CSV columns.
    name: str

    #: Whether larger values are better.
    higher_is_better: bool

    #: During training, run every N epochs.
    #: None = skip during training (only run at final test evaluation).
    run_every_n_epochs: int | None = None

    @classmethod
    def from_data(
        cls,
        data: "DataBundle",
        eval_data: "EvalBundle | None" = None,
    ) -> Self:
        """Build the metric from the run's data bundles.

        Default: `cls()`. Override when the metric needs data-derived
        state (e.g. a tokenizer from `data.extras`).
        """
        return cls()

    def train_loader(
        self,
        data: "DataBundle",
        eval_data: "EvalBundle | None",
    ) -> DataLoader:
        """Loader used during training. Default: data.loader_val.

        Only called when `run_every_n_epochs` is not None.
        """
        return data.loader_val

    def test_loader(
        self,
        data: "DataBundle",
        eval_data: "EvalBundle | None",
    ) -> DataLoader:
        """Loader used after training. Default: data.loader_test."""
        return data.loader_test

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        """Score `model` over `loader`. Returns a single float."""
