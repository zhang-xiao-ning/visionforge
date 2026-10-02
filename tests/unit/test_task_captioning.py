"""Tests for CaptioningTask."""

from typing import Any

import torch
import torch.nn as nn

from tasks.captioning import CaptioningTask


class _DummyCaptioningModel(nn.Module):
    def __init__(self, vocab_size: int = 20) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        return torch.zeros(B, L, self.vocab_size, device=images.device) + self.dummy


def _make_batch(B: int = 2, L: int = 5, image_size: int = 32) -> dict[str, Any]:
    images = torch.randn(B, 3, image_size, image_size)
    input_ids = torch.randint(1, 10, (B, L))
    target_ids = torch.randint(1, 10, (B, L))
    target_ids[:, -1] = -100
    return {"image": images, "input_ids": input_ids, "target_ids": target_ids}


def test_train_step_returns_scalar_loss() -> None:
    task = CaptioningTask()
    model = _DummyCaptioningModel(vocab_size=20)
    batch = _make_batch()
    loss = task.train_step(model, batch, torch.device("cpu"), torch.float32)
    assert loss.shape == ()
    assert loss.item() > 0


def test_padding_positions_are_ignored() -> None:
    task = CaptioningTask()
    model = _DummyCaptioningModel(vocab_size=20)

    batch_a = _make_batch()
    batch_b = {k: v.clone() for k, v in batch_a.items()}
    batch_b["target_ids"][:, -1] = 5

    loss_a = task.train_step(model, batch_a, torch.device("cpu"), torch.float32)
    loss_b = task.train_step(model, batch_b, torch.device("cpu"), torch.float32)

    assert torch.allclose(loss_a, loss_b)
