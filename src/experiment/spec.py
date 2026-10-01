"""Experiment specification: TrainConfig / Stage / Experiment.

Three dataclasses that define "how to run an experiment":

- TrainConfig: training parameters (epochs, lr, optimizer, ...)
- Stage: one phase of multi-stage training (name, freeze, overrides)
- Experiment: registry entry (which model / task / data + defaults)

The registry (`src/registry.py`) holds Experiment objects. Stages are
NOT in the registry — they are provided at run time (CLI or YAML).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn as nn

    from tasks.base import Task


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
    """One phase of multi-stage training.

    `overrides` is a subset of TrainConfig field names. At runtime the
    runner merges `Experiment.config` with `stage.overrides` to produce
    the actual config for this stage.
    """

    name: str
    freeze: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """Registry entry.

    Holds concrete classes (model / task) and the dataset key, plus
    defaults (seed / amp / config). Stages are NOT here.
    """

    model: type[nn.Module]
    task: type[Task]
    data: str
    seed: int = 42
    amp: bool = False
    config: TrainConfig = field(default_factory=TrainConfig)
    category: str = ""
