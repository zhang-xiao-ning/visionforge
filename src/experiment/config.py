"""Training configuration."""

from dataclasses import dataclass

from runtime import DEFAULT_EXPERIMENT


@dataclass
class TrainConfig:
    experiment: str = DEFAULT_EXPERIMENT
    epochs: int = 1
    learning_rate: float = 1e-2
    momentum: float = 0.9
    nesterov: bool = True
    seed: int = 42
    lr_scheduler: str = "none"
    step_size: int = 10
    gamma: float = 0.1
    early_stop_patience: int = 0
    amp: bool = False

    # ---- Optimizer ----
    optimizer: str = "sgd"  # "sgd" / "adamw"
    weight_decay: float = 0.0  # only for adamw

    # ---- LR schedule (step-level when warmup > 0) ----
    warmup_steps: int = 0  # 0 = no warmup

    # ---- Training dynamics ----
    accum_steps: int = 1  # gradient accumulation
    grad_clip: float = 0.0  # 0 = no clipping
