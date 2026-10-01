"""Task interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class Stage:
    """One phase of a multi-stage training run.

    Single-stage tasks (e.g. classification, from-scratch captioning)
    declare `stages = None` and use `TrainConfig.epochs`.

    Multi-stage tasks (e.g. LLaVA: align → instruct) declare an explicit
    list. Each stage runs for `epochs` epochs with the named parameter
    groups frozen.

    Example:
        stages = [
            Stage(name="align", epochs=1, freeze=["vision", "llm_base"]),
            Stage(name="instruct", epochs=2, freeze=["vision"]),
        ]

    Names in `freeze` must match keys from `Model.param_groups()`.
    """

    name: str
    epochs: int
    freeze: list[str] = field(default_factory=list)


class Task(ABC):
    """Encapsulates task-specific logic (loss, metrics, stages).

    A Task does NOT know about:
    - optimizers / schedulers
    - AMP / DDP
    - logging / checkpointing

    It only knows: given a model and a batch, how to compute loss / metrics.
    And, optionally, how the training run is split into stages.
    """

    # Which metric from `eval_step` should be used for best-model selection
    primary_metric: str = "loss"

    # Whether a larger primary_metric is better (True for accuracy,
    # False for loss / perplexity)
    higher_is_better: bool = True

    # Multi-stage training plan. None = single stage, use TrainConfig.epochs.
    stages: list[Stage] | None = None

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
