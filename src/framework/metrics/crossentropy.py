import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from framework.interfaces import Metric, eval_mode


class CrossEntropy(Metric):
    name = "loss"
    higher_is_better = False
    run_every_n_epochs = 1

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        total_loss = 0.0
        total_samples = 0
        with eval_mode(model):
            for batch in loader:
                x, y = batch
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)
                loss_sum = F.cross_entropy(model(x), y, reduction="sum")
                total_loss += loss_sum.item()
                total_samples += y.size(0)
        return total_loss / max(total_samples, 1)
