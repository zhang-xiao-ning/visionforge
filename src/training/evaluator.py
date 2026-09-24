import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from runtime import DTYPE, device
from tasks.base import Task


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task: Task,
) -> dict[str, float]:
    """Aggregate task.eval_step over the loader. Returns averaged metrics."""
    model.eval()

    totals: dict[str, float] = {}
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            metrics = task.eval_step(model, batch, device, DTYPE)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.0) + v
            n_batches += 1

    model.train()

    if n_batches == 0:
        return {k: 0.0 for k in totals}
    return {k: v / n_batches for k, v in totals.items()}
