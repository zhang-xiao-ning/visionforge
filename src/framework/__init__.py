"""visionforge framework API.

Everything re-exported here is meant to be reusable across projects.
Application code (models/, data/, main.py) should import from here,
not from internal modules directly.

This does NOT yet fully decouple framework from application. In
particular, `ExperimentRunner` currently imports application-level
modules (registry, data.datasets). A future refactor can invert that
dependency by injecting the model/loader builders as parameters.

For now, this module makes the intended boundary explicit and prepares
the codebase for a future framework/application split.
"""

from experiment.artifacts import RunArtifacts
from experiment.runner import ExperimentRunner
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
    # Experiment
    "ExperimentRunner",
    "RunArtifacts",
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
