"""Logging with custom levels + CSV/TensorBoard channels.

Levels (from silent to verbose):
    NONE  - silence everything (for benchmarks)
    ERROR - only errors (NaN, OOM)
    INFO  - run-level events (start, save, done)   [default]
    FLOW  - epoch / stage level events
    DEBUG - batch-level details

Control via the VISIONFORGE_LOG_LEVEL environment variable.

Public interface:
    get_logger(name, log_file, csv_path, tb_dir, ...) -> AppLogger
    AppLogger.debug / flow / info / error       (text, single message)
    AppLogger.record(epoch, train_loss, val_metrics, primary_metric, lr)
    AppLogger.close()
    AppLogger.raw                                (stdlib interop)
"""

import csv
import logging
import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

# Custom levels. NONE is above CRITICAL so it silences everything,
# including errors.
NONE = 100
ERROR = logging.ERROR  # 40
INFO = logging.INFO  # 20
FLOW = 21  # just above INFO
DEBUG = logging.DEBUG  # 10

logging.addLevelName(FLOW, "FLOW")
logging.addLevelName(NONE, "NONE")

_LEVELS: dict[str, int] = {
    "none": NONE,
    "error": ERROR,
    "info": INFO,
    "flow": FLOW,
    "debug": DEBUG,
}

DEFAULT_LEVEL = "info"

CSV_HEADER = ["epoch", "train_loss", "val_metric", "lr"]


def parse_level(name: str) -> int:
    """Convert a level name to its numeric value. Raises on unknown name."""
    key = name.lower()
    if key not in _LEVELS:
        raise ValueError(f"Unknown log level '{name}'. Known: {sorted(_LEVELS)}")
    return _LEVELS[key]


class AppLogger:
    """Logger with text, CSV, and TensorBoard channels.

    Text:  use level methods (debug / flow / info / error).
    CSV:   `record()` appends one row per call.
    TB:    `record()` writes scalars per call.

    Only the master process should call `record()` — the caller
    (train hooks) is responsible for the rank check.
    """

    def __init__(
        self,
        name: str,
        log_file: Path,
        csv_path: Path | None = None,
        tb_dir: Path | None = None,
        csv_append: bool = False,
        level: str | None = None,
    ) -> None:
        if level is None:
            level = os.environ.get("VISIONFORGE_LOG_LEVEL", DEFAULT_LEVEL)

        # ---- text channel ----
        self._logger = logging.getLogger(name)
        self._logger.setLevel(parse_level(level))
        self._logger.handlers.clear()
        self._logger.propagate = False

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        self._logger.addHandler(sh)

        # ---- CSV channel ----
        self._csv_path = csv_path
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not (csv_append and csv_path.exists()):
                with open(csv_path, "w", newline="") as f:
                    csv.writer(f).writerow(CSV_HEADER)

        # ---- TensorBoard channel ----
        self._writer = SummaryWriter(log_dir=str(tb_dir)) if tb_dir is not None else None

    # ----- text methods -----

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def flow(self, msg: str) -> None:
        self._logger.log(FLOW, msg)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def __call__(self, msg: str, level: str = DEFAULT_LEVEL) -> None:
        """Log at a named level: logger("...", level="debug")."""
        self._logger.log(parse_level(level), msg)

    # ----- structured record (CSV + TB) -----

    def record(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, float],
        primary_metric: str,
        lr: float,
    ) -> None:
        """Write one epoch of metrics to CSV and TensorBoard."""
        self._record_csv(epoch, train_loss, val_metrics[primary_metric], lr)
        self._record_tb(epoch, train_loss, val_metrics, lr)

    def _record_csv(self, epoch: int, train_loss: float, val_metric: float, lr: float) -> None:
        if self._csv_path is None:
            return
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_metric:.6f}", f"{lr:.6g}"])

    def _record_tb(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict[str, float],
        lr: float,
    ) -> None:
        if self._writer is None:
            return
        self._writer.add_scalar("loss/train", train_loss, epoch)
        for k, v in val_metrics.items():
            self._writer.add_scalar(f"val/{k}", v, epoch)
        self._writer.add_scalar("lr", lr, epoch)

    # ----- lifecycle -----

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        # Close all stdlib handlers to release file descriptors.
        for handler in self._logger.handlers:
            handler.close()
        self._logger.handlers.clear()

    # ----- stdlib interop -----

    @property
    def raw(self) -> logging.Logger:
        """Underlying stdlib logger. For third-party code that needs one."""
        return self._logger


def get_logger(
    name: str,
    log_file: Path,
    csv_path: Path | None = None,
    tb_dir: Path | None = None,
    csv_append: bool = False,
    level: str | None = None,
) -> AppLogger:
    return AppLogger(
        name,
        log_file,
        csv_path=csv_path,
        tb_dir=tb_dir,
        csv_append=csv_append,
        level=level,
    )
