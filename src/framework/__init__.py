"""visionforge framework API.

Framework = training platform. It knows how to train, not what to train.
Application-level orchestration lives outside this package.

Dependency direction:
    application → framework
    framework   ↛ application
"""

# ---- Runtime ----
# ---- Experiment ----
from framework.experiment import build_scheduler
from framework.experiment.artifacts import RunArtifacts
from framework.experiment.loader import load_yaml, parse_stages
from framework.experiment.runner import ExperimentRunner
from framework.experiment.spec import Experiment, RunParams, Stage, TrainConfig

# ---- Inference ----
from framework.inference import SamplingConfig, apply_sampling, sample

# ---- Interfaces ----
from framework.interfaces import (
    DataBundle,
    DataContext,
    EvalBundle,
    Metric,
    Model,
    Task,
    eval_mode,
)

# ---- Metrics ----
from framework.metrics import Accuracy, CrossEntropy, Perplexity, TokenPerplexity, corpus_bleu

# ---- Registry ----
from framework.registry import DATASETS, EXPERIMENTS, DatasetRegistry, ExperimentRegistry
from framework.runtime import DTYPE, PRINT_EVERY, get_device, use_cuda

# ---- Training ----
from framework.training import (
    EpochContext,
    MetricTracker,
    SingleDeviceStrategy,
    StepContext,
    TrainHooks,
    TrainingStrategy,
    build_strategy,
    train,
)

# ---- Utils ----
from framework.utils import (
    CHECKPOINTS_PATH,
    DATASETS_PATH,
    OUTPUTS_PATH,
    AppLogger,
    format_env_info,
    get_env_info,
    get_logger,
    set_seed,
)

__all__ = [
    # runtime
    "DTYPE",
    "PRINT_EVERY",
    "get_device",
    "use_cuda",
    # interfaces
    "DataBundle",
    "DataContext",
    "EvalBundle",
    "Metric",
    "Model",
    "Task",
    "eval_mode",
    # training
    "EpochContext",
    "MetricTracker",
    "SingleDeviceStrategy",
    "StepContext",
    "TrainHooks",
    "TrainingStrategy",
    "build_strategy",
    "train",
    # metrics
    "Accuracy",
    "CrossEntropy",
    "Perplexity",
    "TokenPerplexity",
    "corpus_bleu",
    # experiment
    "RunArtifacts",
    "load_yaml",
    "parse_stages",
    "ExperimentRunner",
    "Experiment",
    "RunParams",
    "Stage",
    "TrainConfig",
    "build_scheduler",
    # registry
    "DATASETS",
    "EXPERIMENTS",
    "DatasetRegistry",
    "ExperimentRegistry",
    # utils
    "CHECKPOINTS_PATH",
    "DATASETS_PATH",
    "OUTPUTS_PATH",
    "AppLogger",
    "format_env_info",
    "get_env_info",
    "get_logger",
    "set_seed",
    # inference
    "SamplingConfig",
    "apply_sampling",
    "sample",
]
