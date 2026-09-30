"""Main training loop.

Single-process semantics: the loop never asks "which process am I?".
Anything that differs between master and worker processes lives in
TrainHooks (supplied by ExperimentRunner).

Entry / exit are logged with the process rank for DDP debugging.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from experiment.config import TrainConfig
from runtime import DTYPE, device
from tasks.base import Task
from training.hooks import EpochContext, StepContext, TrainHooks
from training.strategy import TrainingStrategy


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader_train: DataLoader,
    loader_val: DataLoader,
    task: Task,
    cfg: TrainConfig,
    strategy: TrainingStrategy,
    hooks: TrainHooks,
    scheduler: lr_scheduler.LRScheduler | None = None,
    scheduler_mode: str = "epoch",
    start_epoch: int = 1,
) -> dict[str, float | int]:
    """Run the training loop.

    `model` must already be on the target device (ExperimentRunner wraps it).
    Side effects (logging, evaluation, checkpoint tracking) live in `hooks`.
    """
    if hooks.logger is not None:
        hooks.logger.flow(f"Enter function: train, rank: {strategy.rank}")

    # ---- one-time setup ----
    amp_enabled = cfg.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    last_epoch = start_epoch - 1

    for e in range(start_epoch, cfg.epochs + 1):
        strategy.on_epoch_start(e)
        if hooks.logger is not None:
            hooks.logger.flow(f"Epoch {e} start, rank: {strategy.rank}")

        running_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for t, batch in enumerate(loader_train):
            model.train()

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = task.train_step(model, batch, device, DTYPE)
                loss = loss / cfg.accum_steps

            scaler.scale(loss).backward()

            if (t + 1) % cfg.accum_steps == 0:
                if cfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None and scheduler_mode == "step":
                    scheduler.step()

            unnormalized = loss.item() * cfg.accum_steps
            running_loss += unnormalized
            n_batches += 1
            hooks.on_step(StepContext(epoch=e, step=t, loss=unnormalized))

        avg_loss = running_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        if scheduler is not None and scheduler_mode == "epoch":
            scheduler.step()

        ctx = EpochContext(
            epoch=e,
            avg_loss=avg_loss,
            lr=lr_now,
            model=model,
            loader_val=loader_val,
            task=task,
        )
        should_continue = hooks.on_epoch_end(ctx)

        if hooks.logger is not None:
            hooks.logger.flow(f"Epoch {e} end, rank: {strategy.rank}")
        last_epoch = e
        if not should_continue:
            break

    hooks.on_train_end()
    if hooks.logger is not None:
        hooks.logger.flow(f"Exit function: train, rank: {strategy.rank}")

    return {"last_epoch": last_epoch, **hooks.result()}
