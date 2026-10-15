"""Metrics for self-supervised DINO training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from framework.interfaces import Metric, eval_mode


class StudentSimilarity(Metric):
    """Average pairwise cosine similarity of student outputs on val.

    The DINO head is L2-normalised, so `out @ out.T` is the matrix of
    pairwise cosines. Under collapse every output converges to the
    same direction and the off-diagonal mean goes to 1. Diverse
    representations give a value near 0 (random unit vectors in K dims
    are nearly orthogonal).

    Lower is better. Track this to catch collapse; it is NOT a proxy
    for feature quality — that is what a linear probe is for, in a
    later pass.
    """

    name = "student_sim"
    higher_is_better = False
    run_every_n_epochs = 1

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        total = 0.0
        n_batches = 0
        with eval_mode(model):
            for batch in loader:
                # SSL loaders yield (view1, view2); the metric only
                # needs one view.
                view1 = batch[0] if isinstance(batch, (tuple, list)) else batch
                view1 = view1.to(device=device, dtype=dtype)
                out = model(view1)  # (B, K), L2-normalised
                B = out.size(0)
                if B < 2:
                    continue
                sim = out @ out.T
                off_diag = ~torch.eye(B, dtype=torch.bool, device=device)
                total += sim[off_diag].mean().item()
                n_batches += 1
        return total / max(n_batches, 1)
