"""Task interface.

A Task defines *what to optimize*: given a model and a batch, how to
compute the training loss. It also owns the per-step lifecycle hooks
that some training objectives need (e.g. DINO's EMA teacher update)
and any auxiliary state that must survive a checkpoint/resume cycle.

Metrics are NOT here — they are first-class objects declared per
experiment.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Self

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from framework.interfaces.bundle import DataBundle


class Task(ABC):
    @classmethod
    def from_data(cls, bundle: "DataBundle") -> Self:
        """Build the task from a DataBundle. Default: cls()."""
        return cls()

    def attach(self, model: nn.Module, device: torch.device) -> None:
        """Called once after the model is wrapped and moved to `device`,
        before the optimizer is built.

        Tasks that own auxiliary parameters (e.g. an EMA teacher) build
        them here. Auxiliary parameters MUST NOT appear in
        `model.param_groups()` — they are not trained by the optimizer.

        Default: no-op.
        """
        return

    @abstractmethod
    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute the loss for one batch. Must NOT call backward."""

    def step_end(self, model: nn.Module) -> None:
        """Called after optimizer.step() + optimizer.zero_grad(), i.e.
        only on batches that complete a gradient-accumulation window.

        DINO updates its EMA teacher here. Note: under AMP, if the
        scaler skips the step due to inf/nan gradients, the student
        weights are unchanged and this hook still fires — the teacher
        EMA update is therefore against an unchanged student, which is
        the intended behaviour.

        Default: no-op.
        """
        return

    def state_dict(self) -> dict[str, Any]:
        """Task-owned state for checkpointing.

        Anything returned here is saved and restored across resume.
        Values must be torch-saveable (tensors, primitives, nested
        containers of those). Default: {}.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore task-owned state saved by `state_dict`. Default: no-op."""
        return
