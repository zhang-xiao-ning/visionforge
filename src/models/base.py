"""Setup helpers for experiments whose model construction depends on data.

Only used by tasks where model parameters come from the dataset
(e.g. captioning: vocab_size = tokenizer.vocab_size).

Classification models do NOT need this — they have no data dependency.

This is deliberately duck-typed (no ABC, no base class):
any model class that defines `setup(ctx)` opts in.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from experiment.config import TrainConfig
    from tasks.base import Task


@dataclass
class SetupContext:
    """Inputs shared by any setup() call."""

    cfg: "TrainConfig"
    dataset_name: str
    batch_size: int
    num_train: int | None = None


@dataclass
class ExperimentBundle:
    """What setup() returns: everything the runner needs to start training."""

    model: nn.Module
    task: "Task"
    loader_train: DataLoader
    loader_val: DataLoader
    loader_test: DataLoader
