import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, hidden_size=4000, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)