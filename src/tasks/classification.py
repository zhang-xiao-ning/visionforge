"""Classification task: cross-entropy loss + accuracy."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.base import Task


class ClassificationTask(Task):
    """Standard image classification."""

    primary_metric = "acc"

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        x, y = batch
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        return F.cross_entropy(scores, y)

    def eval_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, float]:
        x, y = batch
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        loss = F.cross_entropy(scores, y)
        preds = scores.argmax(dim=1)
        correct = (preds == y).sum().item()
        return {
            "acc": correct / y.size(0),
            "loss": loss.item(),
        }
