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
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from data.datasets import build_loaders
from experiment.artifacts import RunArtifacts
from experiment.config import TrainConfig
from models.base import SetupContext
from registry import EXPERIMENTS
from runtime import NUM_TRAIN, device
from tasks.classification import ClassificationTask
from training.evaluator import evaluate
from training.strategy import TrainingStrategy, build_strategy
from training.train import train
from utils.env import format_env_info


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model for DDP-wrapped modules."""
    return model.module if hasattr(model, "module") else model


def _build_scheduler(
    optimizer: optim.Optimizer, cfg: TrainConfig
) -> lr_scheduler.LRScheduler | None:
    if cfg.lr_scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return None


class ExperimentRunner:
    """Orchestrates everything that happens during one training run."""

    def __init__(
        self,
        cfg: TrainConfig,
        dataset_name: str,
        batch_size: int,
        strategy: TrainingStrategy | None = None,
        resume_path: str | None = None,
        outputs_dir: Path | None = None,
        checkpoints_dir: Path | None = None,
        num_train: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.resume_path = resume_path
        self.num_train = num_train
        self.strategy = strategy if strategy is not None else build_strategy(device)

        self.artifacts = RunArtifacts.create(
            cfg=cfg,
            dataset_name=dataset_name,
            batch_size=batch_size,
            strategy=self.strategy,
            resume_path=resume_path,
            outputs_dir=outputs_dir,
            checkpoints_dir=checkpoints_dir,
        )

        # TODO: this if-fork is a temporary shape.
        # Captioning needs a data-dependent model (vocab_size from tokenizer),
        # so it exposes a `setup(ctx)` classmethod that builds
        # (model, task, loaders) together. Plain classifiers use the
        # default path below. When a third setup-style model appears,
        # extract this into experiment/builder.py.
        model_cls, _ = EXPERIMENTS[self.cfg.experiment]

        if hasattr(model_cls, "setup"):
            ctx = SetupContext(
                cfg=self.cfg,
                dataset_name=self.dataset_name,
                batch_size=self.batch_size,
                num_train=self.num_train,
            )
            bundle = model_cls.setup(ctx)
            self.model = self.strategy.wrap_model(bundle.model, device)
            self.task = bundle.task
            self.loader_train = bundle.loader_train
            self.loader_val = bundle.loader_val
            self.loader_test = bundle.loader_test
        else:
            self.loader_train, self.loader_val, self.loader_test = build_loaders(
                name=self.dataset_name,
                batch_size=self.batch_size,
                num_train=self.num_train if self.num_train is not None else NUM_TRAIN,
                strategy=self.strategy,
            )
            self.model = self._build_model()
            self.task = ClassificationTask()

        self.optimizer = self._build_optimizer()
        self.scheduler = _build_scheduler(self.optimizer, cfg)
        self.start_epoch, self.best_acc = self._maybe_resume()

    # ---------- construction ----------

    def _build_loaders(self) -> tuple[torch.utils.data.DataLoader, ...]:
        return build_loaders(
            name=self.dataset_name,
            batch_size=self.batch_size,
            num_train=self.num_train if self.num_train is not None else NUM_TRAIN,
            strategy=self.strategy,
        )

    def _build_model(self) -> torch.nn.Module:
        model_cls, _ = EXPERIMENTS[self.cfg.experiment]
        model = model_cls()
        return self.strategy.wrap_model(model, device)

    def _build_optimizer(self) -> optim.Optimizer:
        return optim.SGD(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            momentum=self.cfg.momentum,
            nesterov=self.cfg.nesterov,
        )

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
            self.artifacts.logger.info("Resumed from %s", self.resume_path)
            self.artifacts.logger.info("Resume at epoch %d, best_acc = %.4f", start_epoch, best_acc)

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
        logger.info("Experiment: %s", self.cfg.experiment)
        logger.info("Dataset: %s", self.dataset_name)
        logger.info("Device: %s", device)
        logger.info("Config: %s", self.cfg)
        logger.info(format_env_info())
        logger.info("=" * 60)

    def _evaluate_test(self) -> float:
        if not self.artifacts.is_main:
            return 0.0
        test_metrics = evaluate(self.model, self.loader_test, self.task)
        test_metric = test_metrics[self.task.primary_metric]
        if self.artifacts.logger is not None:
            metrics_str = ", ".join(f"{k} = {v:.4f}" for k, v in test_metrics.items())
            self.artifacts.logger.info("Test %s", metrics_str)
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
            },
            self.artifacts.ckpt_path,
        )
        if self.artifacts.logger is not None:
            self.artifacts.logger.info("Saved to %s", self.artifacts.ckpt_path)
