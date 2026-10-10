# """visionforge framework API.
#
# Framework = training platform. It knows how to train, not what to train.
# Application-level orchestration (ExperimentRunner, RunArtifacts) lives
# outside this package and imports from here — never the other way around.
#
# Dependency direction:
#     application → framework
#     framework   ↛ application
# """
#
# # ---- Runtime ----
# # ---- Experiment ----
# from framework.experiment import (
#     Experiment,
#     ExperimentRunner,
#     RunArtifacts,
#     RunParams,
#     Stage,
#     TrainConfig,
# )
#
# # ---- Interfaces ----
# from framework.interfaces import DataBundle, DataContext, EvalBundle, Metric, Model, Task, eval_mode
#
# # ---- Metrics ----
# from framework.metrics import Accuracy, CrossEntropy, Perplexity, corpus_bleu
#
# # ---- Registry ----
# from framework.registry import EXPERIMENTS, ExperimentRegistry
# from framework.runtime import DTYPE, PRINT_EVERY, get_device, use_cuda
#
# # ---- Training ----
# from framework.training import (
#     EpochContext,
#     MetricTracker,
#     SingleDeviceStrategy,
#     StepContext,
#     TrainHooks,
#     TrainingStrategy,
#     build_strategy,
#     train,
# )
#
# # ---- Utils ----
# from framework.utils import (
#     CHECKPOINTS_PATH,
#     DATASETS_PATH,
#     OUTPUTS_PATH,
#     AppLogger,
#     format_env_info,
#     get_env_info,
#     get_logger,
#     set_seed,
# )
#
# __all__ = [
#     # runtime
#     "DTYPE",
#     "PRINT_EVERY",
#     "get_device",
#     "use_cuda",
#     # interfaces
#     "DataBundle",
#     "DataContext",
#     "EvalBundle",
#     "Metric",
#     "Model",
#     "Task",
#     "eval_mode",
#     # training
#     "EpochContext",
#     "MetricTracker",
#     "SingleDeviceStrategy",
#     "StepContext",
#     "TrainHooks",
#     "TrainingStrategy",
#     "build_strategy",
#     "train",
#     # experiment
#     "RunArtifacts",
#     "ExperimentRunner",
#     "Experiment",
#     "RunParams",
#     "Stage",
#     "TrainConfig",
#     # metrics
#     "Accuracy",
#     "CrossEntropy",
#     "Perplexity",
#     "corpus_bleu",
#     # registry
#     "EXPERIMENTS",
#     "ExperimentRegistry",
#     # utils
#     "CHECKPOINTS_PATH",
#     "DATASETS_PATH",
#     "OUTPUTS_PATH",
#     "AppLogger",
#     "format_env_info",
#     "get_env_info",
#     "get_logger",
#     "set_seed",
# ]
