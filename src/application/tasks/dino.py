"""DINO task: self-distillation with EMA teacher.

Owns everything DINO-specific that the framework's generic loop does
not know about:

- attach(): deep-copies the student into a frozen teacher and creates
  a running centre for the teacher's softmax. Called once by the
  runner before the optimizer is built.
- step_end(): runs after every optimizer step — EMA-updates the
  teacher, updates the centre, and advances the teacher temperature
  schedule.
- state_dict() / load_state_dict(): persist teacher parameters,
  centre and step counter so a resumed run is bit-identical to an
  uninterrupted one.

The framework's contract is exercised end-to-end: this is the first
task that actually uses attach / step_end / state_dict.
"""

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.interfaces import Task

# Default DINO hyperparameters (adapted for CIFAR-10 scale).
EMA_MOMENTUM = 0.996
CENTER_MOMENTUM = 0.9
TEACHER_TEMP_START = 0.04
TEACHER_TEMP_END = 0.07
TEACHER_TEMP_WARMUP_STEPS = 30
STUDENT_TEMP = 0.1


class DINOTask(Task):
    # Teacher lives on the task, not on the student model. Runner builds
    # the optimizer from model.param_groups(), so teacher params are
    # automatically excluded — no extra bookkeeping needed.
    teacher: nn.Module
    center: torch.Tensor
    step_count: int
    _last_teacher_logits: torch.Tensor

    def attach(self, model: nn.Module, device: torch.device) -> None:
        teacher = copy.deepcopy(model)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.to(device)
        teacher.eval()
        self.teacher = teacher

        # Infer K from the teacher's head output dimension.
        out_dim = teacher.head.out_dim
        self.center = torch.zeros(out_dim, device=device)
        self.step_count = 0

    def train_step(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        view1, view2 = batch
        view1 = view1.to(device=device, dtype=dtype)
        view2 = view2.to(device=device, dtype=dtype)

        student_logits = model(view1)

        with torch.no_grad():
            teacher_logits = self.teacher(view2)
            teacher_probs = F.softmax(
                (teacher_logits - self.center) / self._teacher_temp(),
                dim=-1,
            )
            # Cache for step_end's centre update (detach so no graph kept).
            self._last_teacher_logits = teacher_logits.detach()

        student_log_probs = F.log_softmax(student_logits / STUDENT_TEMP, dim=-1)
        return -(teacher_probs * student_log_probs).sum(dim=-1).mean()

    def step_end(self, model: nn.Module) -> None:
        # Contract: called only after at least one train_step in the
        # current accumulation window, so _last_teacher_logits exists.
        # 1) EMA
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), model.parameters(), strict=True):
                pt.mul_(EMA_MOMENTUM).add_(ps.detach(), alpha=1.0 - EMA_MOMENTUM)

        # 2) Centre uses the teacher logits already computed this step.
        with torch.no_grad():
            self.center.mul_(CENTER_MOMENTUM).add_(
                self._last_teacher_logits.mean(dim=0), alpha=1.0 - CENTER_MOMENTUM
            )

        self.step_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "teacher": self.teacher.state_dict(),
            "center": self.center.detach().cpu(),
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.teacher.load_state_dict(state["teacher"])
        self.center = state["center"].to(self.center.device)
        self.step_count = int(state["step_count"])

    # ---------- internal ----------

    def _teacher_temp(self) -> float:
        """Linear warmup from TEACHER_TEMP_START to TEACHER_TEMP_END."""
        if self.step_count >= TEACHER_TEMP_WARMUP_STEPS:
            return TEACHER_TEMP_END
        frac = self.step_count / TEACHER_TEMP_WARMUP_STEPS
        return TEACHER_TEMP_START + frac * (TEACHER_TEMP_END - TEACHER_TEMP_START)
