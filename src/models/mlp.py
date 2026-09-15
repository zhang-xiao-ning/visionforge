import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_size: int = 3 * 32 * 32, hidden_size: int = 4000, num_classes: int = 10
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
