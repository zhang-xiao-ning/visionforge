"""Run artifacts: paths, logger, TensorBoard writer.

This module centralizes everything related to "what a single run produces":

- file paths (log, csv, checkpoint, config snapshot, tensorboard dir)
- the objects that write to them (logger, writer)
- the "main process only" rule under DDP
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experiment.spec import TrainConfig
from training.strategy import TrainingStrategy
from utils.env import get_env_info
from utils.logger import AppLogger, get_logger
from utils.path import CHECKPOINTS_PATH, OUTPUTS_PATH


@dataclass
class RunArtifacts:
    """All per-run artifacts. Only main process creates loggers/writers."""

    base: str
    log_path: Path
    csv_path: Path
    ckpt_path: Path
    cfg_path: Path
    tb_dir: Path
    logger: AppLogger | None
    is_main: bool

    @classmethod
    def create(
        cls,
        experiment_name: str,
        dataset_name: str,
        config: TrainConfig,
        strategy: TrainingStrategy,
        resume_path: str | None = None,
        outputs_dir: Path | None = None,
        checkpoints_dir: Path | None = None,
    ) -> RunArtifacts:
        if outputs_dir is None:
            outputs_dir = OUTPUTS_PATH
        if checkpoints_dir is None:
            checkpoints_dir = CHECKPOINTS_PATH

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{experiment_name}_{timestamp}"

        log_path = outputs_dir / f"{base}.log"
        ckpt_path = checkpoints_dir / f"{base}.pt"
        cfg_path = outputs_dir / f"{base}.json"
        tb_dir = outputs_dir / base

        # Resume 时复用旧 CSV，保持曲线连续
        if resume_path is not None:
            old_base = Path(resume_path).stem
            csv_path = outputs_dir / f"{old_base}.csv"
            csv_append = True
        else:
            csv_path = outputs_dir / f"{base}.csv"
            csv_append = False

        is_main = strategy.is_main_process()

        if is_main:
            outputs_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            logger = get_logger(
                experiment_name, log_path, csv_path=csv_path, tb_dir=tb_dir, csv_append=csv_append
            )
        else:
            outputs_dir.mkdir(parents=True, exist_ok=True)
            worker_log = outputs_dir / f"{base}.rank{strategy.rank}.log"
            logger = get_logger(f"{experiment_name}.rank{strategy.rank}", worker_log)

        artifacts = cls(
            base=base,
            log_path=log_path,
            csv_path=csv_path,
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            tb_dir=tb_dir,
            logger=logger,
            is_main=is_main,
        )

        if is_main:
            artifacts.save_snapshot(
                experiment_name=experiment_name,
                dataset_name=dataset_name,
                config=config,
                resume_path=resume_path,
            )

        return artifacts

    def save_snapshot(
        self,
        experiment_name: str,
        dataset_name: str,
        config: TrainConfig,
        resume_path: str | None,
    ) -> None:
        snapshot = {
            "experiment": experiment_name,
            "dataset": dataset_name,
            "config": dataclasses.asdict(config),
            "env": get_env_info(),
            "resume_from": resume_path,
        }
        with open(self.cfg_path, "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

    def close(self) -> None:
        if self.logger is not None:
            self.logger.close()
