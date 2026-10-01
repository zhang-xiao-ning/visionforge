"""visionforge framework API.

Framework = training platform. It knows how to train, not what to train.
Application-level orchestration (ExperimentRunner, RunArtifacts) lives
outside this package and imports from here — never the other way around.

Dependency direction:
    application → framework
    framework   ↛ application
"""

from training.evaluator import evaluate
from training.hooks import EpochContext, StepContext, TrainHooks
from training.strategy import TrainingStrategy, build_strategy
from training.tracker import MetricTracker
from training.train import train
from utils.env import format_env_info, get_env_info
from utils.logger import AppLogger, get_logger
from utils.path import CHECKPOINTS_PATH, DATASETS_PATH, OUTPUTS_PATH
from utils.seed import set_seed

__all__ = [
    # Training
    "train",
    "evaluate",
    "TrainHooks",
    "StepContext",
    "EpochContext",
    "MetricTracker",
    # Strategy
    "TrainingStrategy",
    "build_strategy",
    # Logger
    "AppLogger",
    "get_logger",
    # Env / utils
    "format_env_info",
    "get_env_info",
    "CHECKPOINTS_PATH",
    "DATASETS_PATH",
    "OUTPUTS_PATH",
    "set_seed",
]
