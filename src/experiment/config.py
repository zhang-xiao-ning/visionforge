"""Training configuration."""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    experiment: str = "mlp"
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
