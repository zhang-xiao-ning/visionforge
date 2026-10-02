"""Built-in metrics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

IGNORE_INDEX = -100


class Accuracy:
    name = "acc"
    higher_is_better = True

    def evaluate(
        self, model: nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype
    ) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        model.train()
        return correct / max(total, 1)


class CrossEntropy:
    name = "loss"
    higher_is_better = False

    def evaluate(
        self, model: nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype
    ) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=torch.long)
                total_loss += F.cross_entropy(model(x), y).item()
                n += 1
        model.train()
        return total_loss / max(n, 1)


class Perplexity:
    name = "perplexity"
    higher_is_better = False

    def evaluate(
        self, model: nn.Module, loader: DataLoader, device: torch.device, dtype: torch.dtype
    ) -> float:
        model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device=device, dtype=dtype)
                input_ids = batch["input_ids"].to(device=device)
                target_ids = batch["target_ids"].to(device=device)
                logits = model(images, input_ids)
                vocab_size = logits.size(-1)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                total_loss += loss.item()
                n += 1
        model.train()
        avg = total_loss / max(n, 1)
        return float(torch.exp(torch.tensor(avg)).item())
