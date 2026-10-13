"""Language modelling task: next-token cross-entropy."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.interfaces import Task


class LMTask(Task):
    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        input_ids, target_ids = batch
        input_ids = input_ids.to(device=device)
        target_ids = target_ids.to(device=device)

        logits = model(input_ids)
        vocab_size = logits.size(-1)
        return F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
        )
