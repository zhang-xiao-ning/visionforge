"""Evaluate a model on a loader using a list of metrics."""

import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.base import Metric
from runtime import DTYPE, device


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    metrics: list[Metric],
) -> dict[str, float]:
    """Run each metric on (model, loader). Returns {metric.name: value}."""
    return {m.name: m.evaluate(model, loader, device, DTYPE) for m in metrics}
