"""Tests for LMTask."""

import torch
import torch.nn as nn

from application.tasks.lm import LMTask


class _DummyLM(nn.Module):
    def __init__(self, vocab: int = 10) -> None:
        super().__init__()
        self.vocab = vocab
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        return torch.zeros(B, L, self.vocab, device=ids.device) + self.dummy


def test_train_step_returns_scalar() -> None:
    task = LMTask()
    model = _DummyLM()
    x = torch.randint(0, 10, (2, 5))
    y = torch.randint(0, 10, (2, 5))
    loss = task.train_step(model, (x, y), torch.device("cpu"), torch.float32)
    assert loss.shape == ()
    assert loss.item() > 0
