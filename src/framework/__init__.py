"""visionforge framework API.

Framework = training platform. It knows how to train, not what to train.
Application-level orchestration (ExperimentRunner, RunArtifacts) lives
outside this package and imports from here — never the other way around.

Dependency direction:
    application → framework
    framework   ↛ application
"""


# __all__ = [
#     # Training
#     "train",
#     "TrainHooks",
#     "StepContext",
#     "EpochContext",
#     "MetricTracker",
#     # Strategy
#     "TrainingStrategy",
#     "build_strategy",
#     # Logger
#     "AppLogger",
#     "get_logger",
#     # Env / utils
#     "format_env_info",
#     "get_env_info",
#     "CHECKPOINTS_PATH",
#     "DATASETS_PATH",
#     "OUTPUTS_PATH",
#     "set_seed",
# ]
