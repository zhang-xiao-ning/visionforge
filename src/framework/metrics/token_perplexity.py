"""Token-level perplexity for language models.

Unlike `Perplexity` (which is captioning-specific and reads
`batch["image"]`), this metric consumes a plain `(input_ids, target_ids)`
batch and calls `model(input_ids)`. Use this for any text-only LM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from framework.interfaces import Metric, eval_mode


class TokenPerplexity(Metric):
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
            for input_ids, target_ids in loader:
                input_ids = input_ids.to(device=device)
                target_ids = target_ids.to(device=device)
                logits = model(input_ids)
                vocab_size = logits.size(-1)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    target_ids.reshape(-1),
                    reduction="sum",
                )
                total_loss += loss_sum.item()
                total_tokens += target_ids.numel()
        avg = total_loss / max(total_tokens, 1)
        return float(torch.exp(torch.tensor(avg)).item())
