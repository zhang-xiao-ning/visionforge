"""Logging with custom levels for ML training.

Levels (from silent to verbose):
    NONE  - silence everything (for benchmarks)
    ERROR - only errors (NaN, OOM)
    INFO  - run-level events (start, save, done)   [default]
    FLOW  - epoch / stage level events
    DEBUG - batch-level details

Control via the VISIONFORGE_LOG_LEVEL environment variable.

Public interface:
    get_logger(name, path) -> AppLogger
    AppLogger.debug / flow / info / error   (single-message)
    AppLogger(msg, level="debug")
    AppLogger.raw                            (stdlib interop)
"""

import csv
import logging
import os
from pathlib import Path

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


def parse_level(name: str) -> int:
    """Convert a level name to its numeric value. Raises on unknown name."""
    key = name.lower()
    if key not in _LEVELS:
        raise ValueError(f"Unknown log level '{name}'. Known: {sorted(_LEVELS)}")
    return _LEVELS[key]


class AppLogger:
    """Logger with explicit level methods for training code.

    Backed by a standard `logging.Logger`. Use the level methods rather
    than the stdlib API — the level names are the stable interface.
    """

    def __init__(
        self,
        name: str,
        log_file: Path,
        level: str | None = None,
    ) -> None:
        if level is None:
            level = os.environ.get("VISIONFORGE_LOG_LEVEL", DEFAULT_LEVEL)

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

    # ----- level methods -----

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def flow(self, msg: str) -> None:
        self._logger.log(FLOW, msg)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    # ----- generic entry point -----

    def __call__(self, msg: str, level: str = DEFAULT_LEVEL) -> None:
        """Log at a named level: logger("...", level="debug")."""
        self._logger.log(parse_level(level), msg)

    # ----- stdlib interop -----

    @property
    def raw(self) -> logging.Logger:
        """Underlying stdlib logger. For third-party code that needs one."""
        return self._logger


def get_logger(name: str, log_file: Path, level: str | None = None) -> AppLogger:
    return AppLogger(name, log_file, level=level)


class CSVRecorder:
    def __init__(self, csv_path: Path, append: bool = False) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if append and self.csv_path.exists():
            return

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_metric", "lr"])

    def log(
        self, epoch: int, train_loss: float, val_metric: float, lr: float | None = None
    ) -> None:
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_metric:.6f}",
                    f"{lr:.6g}" if lr is not None else "",
                ]
            )
