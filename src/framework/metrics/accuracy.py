import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from framework.interfaces import Metric, eval_mode


class Accuracy(Metric):
    name = "acc"
    higher_is_better = True
    run_every_n_epochs = 1

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        correct = 0
        total = 0
        with eval_mode(model):
            for batch in loader:
                x, y = batch
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1)
