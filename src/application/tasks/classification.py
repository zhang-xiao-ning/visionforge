"""Classification task: cross-entropy loss."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.interfaces import Task


class ClassificationTask(Task):
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
        return F.cross_entropy(model(x), y)
