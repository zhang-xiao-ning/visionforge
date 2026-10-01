"""Command-line interface: argparse → Experiment + TrainConfig."""

import argparse
import dataclasses

from experiment.spec import TrainConfig
from registry import EXPERIMENTS
from runtime import DEFAULT_EXPERIMENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visionforge models.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT,
        choices=list(EXPERIMENTS.keys()),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)

    # TrainConfig overrides — default None so we can distinguish
    # "user did not pass it" from "user passed a value".
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adamw"])
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--no-nesterov", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--lr-scheduler", type=str, default=None, choices=["none", "step", "cosine"]
    )
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--accum-steps", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)

    return parser.parse_args()


# argparse arg name → TrainConfig field name
_CONFIG_FIELDS = {
    "epochs": "epochs",
    "batch_size": "batch_size",
    "optimizer": "optimizer",
    "learning_rate": "learning_rate",
    "momentum": "momentum",
    "weight_decay": "weight_decay",
    "lr_scheduler": "lr_scheduler",
    "step_size": "step_size",
    "gamma": "gamma",
    "warmup_steps": "warmup_steps",
    "accum_steps": "accum_steps",
    "grad_clip": "grad_clip",
    "early_stop_patience": "early_stop_patience",
}


def build_run(
    args: argparse.Namespace,
) -> tuple[str, TrainConfig, int, bool, int | None]:
    """Merge CLI args into the Experiment's default config.

    Returns (experiment_name, config, seed, amp, num_train).
    """
    experiment = EXPERIMENTS[args.experiment]

    overrides: dict[str, object] = {}
    for arg_name, field_name in _CONFIG_FIELDS.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    if args.no_nesterov:
        overrides["nesterov"] = False

    config = dataclasses.replace(experiment.config, **overrides)

    seed = args.seed if args.seed is not None else experiment.seed
    amp = args.amp or experiment.amp

    return args.experiment, config, seed, amp, args.num_train
