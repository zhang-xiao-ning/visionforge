"""Built-in metrics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from framework.interfaces import Matric, eval_mode

IGNORE_INDEX = -100


class Accuracy(Matric):
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


class CrossEntropy(Matric):
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


class Perplexity(Matric):
    name = "perplexity"
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
        total_tokens = 0
        with eval_mode(model):
            for batch in loader:
                images = batch["image"].to(device=device, dtype=dtype)
                input_ids = batch["input_ids"].to(device=device)
                target_ids = batch["target_ids"].to(device=device)
                logits = model(images, input_ids)
                vocab_size = logits.size(-1)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
                # ignore_index positions don't contribute to sum, but
                # they also shouldn't be in the denominator.
                n_valid = (target_ids != IGNORE_INDEX).sum().item()
                total_loss += loss_sum.item()
                total_tokens += n_valid
        avg = total_loss / max(total_tokens, 1)
        return float(torch.exp(torch.tensor(avg)).item())
