"""Captioning task: cross-entropy loss over tokens."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.interfaces import Task

IGNORE_INDEX = -100


class CaptioningTask(Task):
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

        logits = model(images, input_ids)
        vocab_size = logits.size(-1)

        return F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
