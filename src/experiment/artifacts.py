"""Run artifacts: paths, logger, CSV recorder, TensorBoard writer.

This module centralizes everything related to "what a single run produces":

- file paths (log, csv, checkpoint, config snapshot, tensorboard dir)
- the objects that write to them (logger, recorder, writer)
- the "main process only" rule under DDP

Adding a new artifact (e.g. WandB) means extending this class, without
touching training code.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from experiment.config import TrainConfig
from training.strategy import TrainingStrategy
from utils.env import get_env_info
from utils.logger import CSVRecorder, get_logger
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
    logger: logging.Logger | None
    recorder: CSVRecorder | None
    writer: SummaryWriter | None
    is_main: bool

    @classmethod
    def create(
        cls,
        cfg: TrainConfig,
        dataset_name: str,
        batch_size: int,
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
        base = f"{cfg.experiment}_{timestamp}"

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
            logger = get_logger(cfg.experiment, log_path)
            recorder = CSVRecorder(csv_path, append=csv_append)
            writer = SummaryWriter(log_dir=str(tb_dir))
        else:
            logger = None
            recorder = None
            writer = None

        artifacts = cls(
            base=base,
            log_path=log_path,
            csv_path=csv_path,
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            tb_dir=tb_dir,
            logger=logger,
            recorder=recorder,
            writer=writer,
            is_main=is_main,
        )

        if is_main:
            artifacts.save_snapshot(
                cfg=cfg,
                dataset_name=dataset_name,
                batch_size=batch_size,
                resume_path=resume_path,
            )

        return artifacts

    def save_snapshot(
        self,
        cfg: TrainConfig,
        dataset_name: str,
        batch_size: int,
        resume_path: str | None,
    ) -> None:
        snapshot = {
            "config": dataclasses.asdict(cfg),
            "dataset": dataset_name,
            "batch_size": batch_size,
            "env": get_env_info(),
            "resume_from": resume_path,
        }
        with open(self.cfg_path, "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
