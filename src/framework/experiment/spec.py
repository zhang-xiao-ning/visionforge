"""Experiment specification: TrainConfig / Stage / Experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

    from framework.interfaces import Metric, Task


@dataclass
class TrainConfig:
    """Training parameters for one stage (or a single-stage run)."""

    # ---- schedule ----
    epochs: int = 1
    batch_size: int = 64

    # ---- optimizer ----
    optimizer: str = "sgd"
    learning_rate: float = 1e-2
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 0.0

    # ---- LR schedule ----
    lr_scheduler: str = "none"
    step_size: int = 10
    gamma: float = 0.1
    warmup_steps: int = 0

    # ---- training dynamics ----
    accum_steps: int = 1
    grad_clip: float = 0.0
    early_stop_patience: int = 0


@dataclass
class Stage:
    """One phase of multi-stage training."""

    name: str
    freeze: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """Registry entry.

    Holds concrete classes (model / task), the dataset key, the metric
    instances used for evaluation, plus defaults (seed / amp / config).
    Stages are NOT here.
    """

    model: type[nn.Module]
    task: type[Task]
    data: str
    metrics: list[Metric]
    primary_metric: str
    seed: int = 42
    amp: bool = False
    config: TrainConfig = field(default_factory=TrainConfig)
    category: str = ""


@dataclass
class RunParams:
    """What `build_run` produces: everything `main` needs to start a run."""

    experiment_name: str
    config: TrainConfig
    seed: int
    amp: bool
    num_train: int | None
    stages: list[Stage] | None
