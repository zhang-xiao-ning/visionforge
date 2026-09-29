"""ExperimentRunner: orchestrates a single training run.

This class owns everything between "I have a TrainConfig" and
"I have a trained model + a checkpoint on disk":

- paths and loggers/writers (via RunArtifacts)
- data loaders
- model / optimizer / scheduler construction
- optional resume from checkpoint
- the training loop
- test evaluation
- checkpoint saving

Under DDP, only the main process writes logs/checkpoints. The actual
"what differs between topologies" is delegated to a TrainingStrategy.
"""

import dataclasses
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from data.bundle import DataContext
from data.datasets import build_data
from experiment.artifacts import RunArtifacts
from experiment.config import TrainConfig
from registry import EXPERIMENTS
from runtime import device
from training.evaluator import evaluate
from training.strategy import TrainingStrategy, build_strategy
from training.train import train
from utils.env import format_env_info


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model for DDP-wrapped modules."""
    return model.module if hasattr(model, "module") else model


def _build_scheduler(
    optimizer: optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> tuple[lr_scheduler.LRScheduler | None, str]:
    """Build a scheduler. Returns (scheduler, mode).

    mode = "step": scheduler.step() is called every batch
    mode = "epoch": scheduler.step() is called every epoch

    When `warmup_steps > 0`, we use a step-level LambdaLR that combines
    warmup + (optional) cosine decay. Otherwise, the classic epoch-level
    schedulers are used.
    """
    total_steps = cfg.epochs * steps_per_epoch

    # Step-level path: warmup (+ optional cosine)
    if cfg.warmup_steps > 0:

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            if cfg.lr_scheduler == "cosine":
                progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return 1.0

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"

    # Epoch-level path (existing behavior)
    if cfg.lr_scheduler == "step":
        return (
            optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma),
            "epoch",
        )
    if cfg.lr_scheduler == "cosine":
        return (
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs),
            "epoch",
        )
    return None, "epoch"


class ExperimentRunner:
    """Orchestrates everything that happens during one training run."""

    def __init__(
        self,
        cfg: TrainConfig,
        batch_size: int,
        strategy: TrainingStrategy | None = None,
        resume_path: str | None = None,
        outputs_dir: Path | None = None,
        checkpoints_dir: Path | None = None,
        num_train: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.batch_size = batch_size
        self.resume_path = resume_path
        self.num_train = num_train
        self.strategy = strategy if strategy is not None else build_strategy(device)

        exp = EXPERIMENTS[cfg.experiment]
        self.dataset_name = exp["data"]

        self.artifacts = RunArtifacts.create(
            cfg=cfg,
            dataset_name=self.dataset_name,
            batch_size=batch_size,
            strategy=self.strategy,
            resume_path=resume_path,
            outputs_dir=outputs_dir,
            checkpoints_dir=checkpoints_dir,
        )

        # Data first: build the bundle
        ctx = DataContext(
            batch_size=batch_size,
            num_train=num_train,
            strategy=self.strategy,
        )
        self.bundle = build_data(self.dataset_name, ctx)

        # Model and task adapt themselves to the data
        raw_model = exp["model"].from_data(self.bundle)
        self.model = self.strategy.wrap_model(raw_model, device)
        self.task = exp["task"].from_data(self.bundle)

        self.loader_train = self.bundle.loader_train
        self.loader_val = self.bundle.loader_val
        self.loader_test = self.bundle.loader_test

        self.optimizer = self._build_optimizer()
        self.scheduler, self.scheduler_mode = _build_scheduler(
            self.optimizer, cfg, steps_per_epoch=len(self.loader_train)
        )
        self.start_epoch, self.best_acc = self._maybe_resume()

    # ---------- construction ----------

    def _build_optimizer(self) -> optim.Optimizer:
        if self.cfg.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
        if self.cfg.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
            )
        raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    def _maybe_resume(self) -> tuple[int, float]:
        if self.resume_path is None:
            return 1, 0.0

        ckpt = torch.load(self.resume_path, map_location=device)
        _unwrap_model(self.model).load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]

        if self.artifacts.is_main and self.artifacts.logger is not None:
            self.artifacts.logger.info(f"Resumed from {self.resume_path}")
            self.artifacts.logger.info(f"Resume at epoch {start_epoch}, best_acc = {best_acc:.4f}")

        return start_epoch, best_acc

    # ---------- main flow ----------

    def run(self) -> dict[str, float]:
        self._log_header()

        result = train(
            self.model,
            self.optimizer,
            self.loader_train,
            self.loader_val,
            self.task,
            epochs=self.cfg.epochs,
            scheduler=self.scheduler,
            scheduler_mode=self.scheduler_mode,
            accum_steps=self.cfg.accum_steps,
            grad_clip=self.cfg.grad_clip,
            early_stop_patience=self.cfg.early_stop_patience,
            start_epoch=self.start_epoch,
            best_acc=self.best_acc,
            logger=self.artifacts.logger,
            recorder=self.artifacts.recorder,
            use_amp=self.cfg.amp,
            writer=self.artifacts.writer,
            strategy=self.strategy,
        )

        test_acc = self._evaluate_test()
        self._save_checkpoint(result)

        return {"test_acc": test_acc, **result}

    def cleanup(self) -> None:
        self.artifacts.close()
        self.strategy.cleanup()

    # ---------- helpers ----------

    def _log_header(self) -> None:
        if not self.artifacts.is_main or self.artifacts.logger is None:
            return
        logger = self.artifacts.logger
        logger.info("=" * 60)
        logger.info(f"Experiment: {self.cfg.experiment}")
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Config: {self.cfg}")
        logger.info(format_env_info())
        logger.info("=" * 60)

    def _evaluate_test(self) -> float:
        if not self.artifacts.is_main:
            return 0.0
        test_metrics = evaluate(self.model, self.loader_test, self.task)
        test_metric = test_metrics[self.task.primary_metric]
        if self.artifacts.logger is not None:
            metrics_str = ", ".join(f"{k} = {v:.4f}" for k, v in test_metrics.items())
            self.artifacts.logger.info(f"Test {metrics_str}")
        return test_metric

    def _save_checkpoint(self, result: dict[str, float | int]) -> None:
        if not self.artifacts.is_main:
            return
        torch.save(
            {
                "model": _unwrap_model(self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
                "epoch": result["last_epoch"],
                "best_acc": result["best_acc"],
                "config": dataclasses.asdict(self.cfg),
                # Self-contained reconstruction info
                "model_init": self.bundle.model_init,
                "extras": self.bundle.extras,
            },
            self.artifacts.ckpt_path,
        )
        if self.artifacts.logger is not None:
            self.artifacts.logger.info(f"Saved to {self.artifacts.ckpt_path}")
