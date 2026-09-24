import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from runtime import DTYPE, PRINT_EVERY, device
from tasks.base import Task
from training.evaluator import evaluate
from training.strategy import TrainingStrategy
from utils.logger import CSVRecorder


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader_train: DataLoader,
    loader_val: DataLoader,
    task: Task,
    epochs: int = 1,
    scheduler: lr_scheduler.LRScheduler | None = None,
    early_stop_patience: int = 0,
    start_epoch: int = 1,
    best_acc: float = 0.0,
    logger: logging.Logger | None = None,
    recorder: CSVRecorder | None = None,
    use_amp: bool = False,
    writer: SummaryWriter | None = None,
    strategy: TrainingStrategy | None = None,
) -> dict[str, float | int]:
    model = model.to(device=device)
    best_state: dict[str, torch.Tensor] | None = None
    epochs_no_improve = 0
    last_epoch = start_epoch - 1

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def is_main() -> bool:
        return strategy is None or strategy.is_main_process()

    def log(msg: str) -> None:
        if not is_main():
            return
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    for e in range(start_epoch, epochs + 1):
        if strategy is not None:
            strategy.set_epoch(e)

        running_loss = 0.0
        n_batches = 0

        for t, batch in enumerate(loader_train):
            model.train()
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = task.train_step(model, batch, device, DTYPE)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            if t % PRINT_EVERY == 0:
                log(f"Epoch {e}, Iter {t}, loss = {loss.item():.4f}")

        avg_loss = running_loss / max(n_batches, 1)
        lr_now = optimizer.param_groups[0]["lr"]

        if is_main():
            val_metrics = evaluate(model, loader_val, task)
        else:
            val_metrics = {task.primary_metric: 0.0}

        val_metric = val_metrics[task.primary_metric]
        metrics_str = ", ".join(f"{k} = {v:.4f}" for k, v in val_metrics.items())
        log(f"Epoch {e} done. avg_train_loss = {avg_loss:.4f}, {metrics_str}")

        if recorder is not None and is_main():
            recorder.log(e, avg_loss, val_metric, lr_now)

        if writer is not None and is_main():
            writer.add_scalar("loss/train", avg_loss, e)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, e)
            writer.add_scalar("lr", lr_now, e)

        if scheduler is not None:
            scheduler.step()
            log(f"  LR -> {optimizer.param_groups[0]['lr']:.6g}")

        if is_main():
            if val_metric > best_acc:
                best_acc = val_metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                    log(f"Early stopping at epoch {e} (no improve for {epochs_no_improve} epochs)")
                    last_epoch = e
                    break

        last_epoch = e

    if best_state is not None and is_main():
        model.load_state_dict(best_state)

    log(f"Best {task.primary_metric} = {best_acc:.4f}")
    return {"best_acc": best_acc, "last_epoch": last_epoch}
