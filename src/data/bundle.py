"""DataBundle: everything the runner needs from a dataset.

Bundles "data" and "construction info" together so that models and tasks
can be built from the dataset itself (`model_cls.from_data(bundle)`).

- loaders: what training iterates over
- model_init: kwargs to construct the model (num_classes, vocab_size, ...)
- extras: serializable reconstruction hints (tokenizer_name, ...)
"""

from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader

from training.strategy import TrainingStrategy


@dataclass
class DataContext:
    """Inputs for building a dataset."""

    batch_size: int
    num_train: int | None = None
    strategy: TrainingStrategy | None = None


@dataclass
class DataBundle:
    """Everything needed to start a run."""

    loader_train: DataLoader
    loader_val: DataLoader
    loader_test: DataLoader
    model_init: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalBundle:
    """Everything needed for post-training evaluation.

    Separate from DataBundle because evaluation may need a different
    item granularity (e.g. one image + all its references) than training
    (one image + one caption). A dataset that does not need this returns
    None.
    """

    loader: DataLoader
    extras: dict[str, Any] = field(default_factory=dict)
