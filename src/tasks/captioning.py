"""Captioning task: cross-entropy + perplexity."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tasks.base import Task

IGNORE_INDEX = -100  # 文件顶部加常量


class CaptioningTask(Task):
    """Image captioning with cross-entropy over tokens.

    The dataset is responsible for the teacher-forcing shift:

        input_ids  = [BOS, c1, ..., c_{N-1}]
        target_ids = [c1, ..., c_N, EOS]

    We compute cross-entropy with `ignore_index=pad_id` so padding
    positions do not contribute to the loss.
    """

    primary_metric = "perplexity"
    higher_is_better = False  # lower perplexity is better

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        images = batch["image"].to(device=device, dtype=dtype)
        input_ids = batch["input_ids"].to(device=device)
        target_ids = batch["target_ids"].to(device=device)

        logits = model(images, input_ids)  # (B, L, V)
        vocab_size = logits.size(-1)

        return F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def eval_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, float]:
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
        loss_value = loss.item()

        return {
            "loss": loss_value,
            "perplexity": float(torch.exp(loss).item()),
        }
