"""MLP baseline: flatten + two-layer perceptron.

Minimal classifier kept around for two reasons:
- regression tests need a fast, stable, low-bar baseline
- Docker's default training command should finish in seconds

Subclasses Model so that Model.from_data / param_groups / initialize
defaults apply, and so that contract tests pick it up automatically.
"""

import torch
import torch.nn as nn

from framework.interfaces import Model


class MLP(Model):
    def __init__(
        self,
        input_size: int = 3 * 32 * 32,
        hidden_size: int = 4000,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
