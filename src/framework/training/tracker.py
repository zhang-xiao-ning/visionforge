"""Track best metric and early stopping.

Kept separate from the training loop: `train` only calls
`tracker.update(...)`; whether that means "snapshot model + reset
patience" or "increment patience" is the tracker's business.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MetricTracker:
    """Tracks the best metric value and its model snapshot.

    Also implements early stopping: `update` returns False when
    patience is exceeded.
    """

    def __init__(
        self,
        higher_is_better: bool,
        patience: int = 0,
        initial_best: float | None = None,
    ) -> None:
        self.higher_is_better = higher_is_better
        self.patience = patience
        self.best: float = (
            initial_best
            if initial_best is not None
            else (float("-inf") if higher_is_better else float("inf"))
        )
        self.best_state: dict[str, torch.Tensor] | None = None
        self.epochs_no_improve = 0

    def is_improvement(self, value: float) -> bool:
        if self.higher_is_better:
            return value > self.best
        return value < self.best

    def update(self, value: float, model: nn.Module) -> bool:
        """Update tracker. Returns True to continue, False to stop."""
        if self.is_improvement(value):
            self.best = value
            # Snapshot to CPU to avoid a second GPU copy.
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.epochs_no_improve = 0
            return True

        self.epochs_no_improve += 1
        if self.patience > 0 and self.epochs_no_improve >= self.patience:
            return False
        return True

    def restore(self, model: nn.Module) -> None:
        """Restore the best weights into the model (no-op if none)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
