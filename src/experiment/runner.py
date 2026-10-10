"""ExperimentRunner: orchestrates a single training run."""

import dataclasses
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from data.datasets import build_data
from experiment.artifacts import RunArtifacts
from experiment.spec import Stage, TrainConfig
from framework.interfaces import DataContext, Metric
from registry import EXPERIMENTS
from runtime import DTYPE, PRINT_EVERY
from training.hooks import EpochContext, StepContext, TrainHooks
from training.strategy import TrainingStrategy, build_strategy
from training.tracker import MetricTracker
from training.train import LoopConfig, train
from utils.env import format_env_info


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _build_scheduler(
    optimizer: optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> tuple[lr_scheduler.LRScheduler | None, str]:
    # Optimizer steps happen every `accum_steps` batches. Batches at the
    # tail that don't complete an accumulation window are dropped, matching
    # train.py's `(t + 1) % accum_steps == 0` condition.
    opt_steps_per_epoch = max(1, steps_per_epoch // cfg.accum_steps)
    total_steps = cfg.epochs * opt_steps_per_epoch

    if cfg.warmup_steps > 0:

        def lr_lambda(step: int) -> float:
            if step < cfg.warmup_steps:
                return step / max(1, cfg.warmup_steps)
            if cfg.lr_scheduler == "cosine":
                progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return 1.0

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "step"

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
    def __init__(
        self,
        experiment_name: str,
        config: TrainConfig,
        stages: list[Stage] | None = None,
        amp: bool = False,
        strategy: TrainingStrategy | None = None,
        resume_path: str | None = None,
        outputs_dir: Path | None = None,
        checkpoints_dir: Path | None = None,
        num_train: int | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.cfg = config
        self.stages = stages
        self.amp = amp
        self.resume_path = resume_path
        self.num_train = num_train
        self.strategy = strategy if strategy is not None else build_strategy()

        experiment = EXPERIMENTS[experiment_name]
        self.dataset_name = experiment.data
        self.primary_metric = experiment.primary_metric

        self.artifacts = RunArtifacts.create(
            experiment_name=experiment_name,
            dataset_name=self.dataset_name,
            config=config,
            strategy=self.strategy,
            resume_path=resume_path,
            outputs_dir=outputs_dir,
            checkpoints_dir=checkpoints_dir,
        )

        ctx = DataContext(
            batch_size=config.batch_size,
            num_train=num_train,
            strategy=self.strategy,
        )
        self.bundle, self.eval_bundle = build_data(self.dataset_name, ctx)

        raw_model = experiment.model.from_data(self.bundle)
        self.model = self.strategy.wrap_model(raw_model)
        self.task = experiment.task.from_data(self.bundle)
        self.metrics: list[Metric] = [
            m.from_data(self.bundle, self.eval_bundle) for m in experiment.metrics
        ]

        self.loader_train = self.bundle.loader_train
        self.loader_val = self.bundle.loader_val
        self.loader_test = self.bundle.loader_test

        self.optimizer = self._build_optimizer()
        self.scheduler, self.scheduler_mode = _build_scheduler(
            self.optimizer, config, steps_per_epoch=len(self.loader_train)
        )
        self.start_epoch, self.best_acc = self._maybe_resume()
        self._validate_metrics()
        self._tracker = self._build_tracker()

    # ---------- construction ----------
    def _validate_metrics(self) -> None:
        """Ensure primary_metric runs every epoch during training.

        Otherwise early stop / best-model selection has no data on
        some epochs.
        """
        primary = next((m for m in self.metrics if m.name == self.primary_metric), None)
        if primary is None:
            raise ValueError(
                f"primary_metric '{self.primary_metric}' not in metrics: "
                f"{[m.name for m in self.metrics]}"
            )
        if primary.run_every_n_epochs != 1:
            raise ValueError(
                f"primary_metric '{primary.name}' must run every epoch "
                f"(got run_every_n_epochs={primary.run_every_n_epochs})"
            )

    def _primary_higher_is_better(self) -> bool:
        for m in self.metrics:
            if m.name == self.primary_metric:
                return m.higher_is_better
        raise ValueError(
            f"primary_metric '{self.primary_metric}' not in metrics: "
            f"{[m.name for m in self.metrics]}"
        )

    def _build_tracker(self) -> MetricTracker:
        return MetricTracker(
            higher_is_better=self._primary_higher_is_better(),
            patience=self.cfg.early_stop_patience,
            initial_best=self.best_acc if self.resume_path else None,
        )

    def _build_hooks(self) -> TrainHooks:
        """Build hooks. This is the ONLY place that branches on rank."""
        if not self.strategy.is_main_process():
            return TrainHooks(logger=self.artifacts.logger)

        logger = self.artifacts.logger
        tracker = self._tracker
        metrics = self.metrics
        primary_metric = self.primary_metric

        def on_step(ctx: StepContext) -> None:
            if ctx.step % PRINT_EVERY == 0:
                logger.debug(f"Epoch {ctx.epoch}, Iter {ctx.step}, loss = {ctx.loss:.4f}")

        def on_epoch_end(ctx: EpochContext) -> bool:
            active = [
                m
                for m in metrics
                if m.run_every_n_epochs is not None and ctx.epoch % m.run_every_n_epochs == 0
            ]
            val_metrics = {
                m.name: m.evaluate(
                    ctx.model,
                    m.train_loader(self.bundle, self.eval_bundle),
                    self.strategy.device,
                    DTYPE,
                )
                for m in active
            }
            ctx.val_metrics = val_metrics
            primary = val_metrics[primary_metric]
            metrics_str = ", ".join(f"{k} = {v:.4f}" for k, v in val_metrics.items())
            logger.flow(
                f"Epoch {ctx.epoch} done. avg_train_loss = {ctx.avg_loss:.4f}, {metrics_str}"
            )
            logger.record(ctx.epoch, ctx.avg_loss, val_metrics, primary_metric, ctx.lr)
            return tracker.update(primary, ctx.model)

        def on_train_end() -> None:
            logger.info(f"Best {primary_metric} = {tracker.best:.4f}")

        def result() -> dict[str, float]:
            return {"best_acc": tracker.best}

        return TrainHooks(
            logger=logger,
            on_step=on_step,
            on_epoch_end=on_epoch_end,
            on_train_end=on_train_end,
            result=result,
        )

    def _build_optimizer(self) -> optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
        if self.cfg.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.cfg.learning_rate,
                momentum=self.cfg.momentum,
                nesterov=self.cfg.nesterov,
            )
        raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    def _maybe_resume(self) -> tuple[int, float]:
        if self.resume_path is None:
            return 1, 0.0

        ckpt = torch.load(self.resume_path, map_location="cpu")
        _unwrap_model(self.model).load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]

        if self.strategy.is_main_process():
            self.artifacts.logger.info(f"Resumed from {self.resume_path}")
            self.artifacts.logger.info(f"Resume at epoch {start_epoch}, best_acc = {best_acc:.4f}")

        return start_epoch, best_acc

    @staticmethod
    def _to_loop_config(cfg: TrainConfig) -> LoopConfig:
        """Application config → framework config. The only adaptor."""
        return LoopConfig(
            epochs=cfg.epochs,
            accum_steps=cfg.accum_steps,
            grad_clip=cfg.grad_clip,
        )

    # ---------- main flow ----------

    def run(self) -> dict[str, float]:
        self._log_header()
        if self.stages is None:
            result = self._run_single_stage()
        else:
            result = self._run_multi_stage()
        self._tracker.restore(self.model)
        test_acc = self._evaluate_test()
        self._save_checkpoint(result)

        return {"test_acc": test_acc, **result}

    def _run_single_stage(self) -> dict[str, float | int]:
        hooks = self._build_hooks()
        return train(
            self.model,
            self.optimizer,
            self.loader_train,
            self.loader_val,
            self.task,
            self._to_loop_config(self.cfg),
            self.strategy,
            hooks,
            scheduler=self.scheduler,
            scheduler_mode=self.scheduler_mode,
            start_epoch=self.start_epoch,
            amp=self.amp,
        )

    def _run_multi_stage(self) -> dict[str, float | int]:
        assert self.stages is not None
        if self.resume_path is not None:
            raise NotImplementedError(
                "Resume is not supported for multi-stage training. "
                "Run from scratch, or resume a single-stage experiment."
            )

        stages = self.stages
        result: dict[str, float | int] = {}
        for i, stage in enumerate(stages, start=1):
            self.artifacts.logger.flow(
                f"Stage {i}/{len(stages)}: {stage.name} "
                f"(freeze={stage.freeze}, overrides={stage.overrides})"
            )

            self._apply_freeze(stage.freeze)
            self.optimizer = self._build_optimizer()

            stage_cfg = dataclasses.replace(self.cfg, **stage.overrides)
            self.scheduler, self.scheduler_mode = _build_scheduler(
                self.optimizer, stage_cfg, steps_per_epoch=len(self.loader_train)
            )

            self._tracker = self._build_tracker()
            hooks = self._build_hooks()

            result = train(
                self.model,
                self.optimizer,
                self.loader_train,
                self.loader_val,
                self.task,
                self._to_loop_config(stage_cfg),
                self.strategy,
                hooks,
                scheduler=self.scheduler,
                scheduler_mode=self.scheduler_mode,
                start_epoch=1,
                amp=self.amp,
            )

        return result

    def _apply_freeze(self, freeze: list[str]) -> None:
        model = _unwrap_model(self.model)
        groups = model.param_groups()

        unknown = set(freeze) - set(groups.keys())
        if unknown:
            raise ValueError(
                f"Unknown freeze groups: {sorted(unknown)}. Available: {sorted(groups.keys())}"
            )

        for name, params in groups.items():
            should_train = name not in freeze
            for p in params:
                p.requires_grad = should_train

    def cleanup(self) -> None:
        self.artifacts.close()
        self.strategy.cleanup()

    # ---------- helpers ----------

    def _log_header(self) -> None:
        if not self.strategy.is_main_process():
            return
        logger = self.artifacts.logger
        logger.info("=" * 60)
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Device: {self.strategy.device}")
        logger.info(f"Config: {self.cfg}")
        logger.info(format_env_info())
        logger.info("=" * 60)

    def _evaluate_test(self) -> float:
        if not self.strategy.is_main_process():
            return 0.0
        test_metrics = {
            m.name: m.evaluate(
                self.model,
                m.test_loader(self.bundle, self.eval_bundle),
                self.strategy.device,
                DTYPE,
            )
            for m in self.metrics
        }
        test_metric = test_metrics[self.primary_metric]
        metrics_str = ", ".join(f"{k} = {v:.4f}" for k, v in test_metrics.items())
        self.artifacts.logger.info(f"Test {metrics_str}")
        return test_metric

    def _save_checkpoint(self, result: dict[str, float | int]) -> None:
        if not self.strategy.is_main_process():
            return
        self.strategy.save(
            self.artifacts.ckpt_path,
            {
                "model": _unwrap_model(self.model).state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (self.scheduler.state_dict() if self.scheduler is not None else None),
                "epoch": result["last_epoch"],
                "best_acc": result["best_acc"],
                "config": dataclasses.asdict(self.cfg),
                "model_init": self.bundle.model_init,
                "extras": self.bundle.extras,
            },
        )
        self.artifacts.logger.info(f"Saved to {self.artifacts.ckpt_path}")
