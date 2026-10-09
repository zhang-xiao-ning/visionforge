"""Command-line interface: argparse + optional YAML → run parameters."""

import argparse
import dataclasses
from pathlib import Path
from typing import Any

from experiment.loader import (
    coerce_overrides,
    load_yaml,
    parse_stages,
    split_overrides,
    validate_override_keys,
)
from experiment.spec import RunParams, Stage
from registry import EXPERIMENTS
from runtime import DEFAULT_EXPERIMENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visionforge models.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file. Mutually exclusive with --experiment.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment name (looked up in the registry).",
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


# argparse dest → TrainConfig field name
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


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Extract non-None CLI args as TrainConfig overrides."""
    overrides: dict[str, Any] = {}
    for arg_name, field_name in _CONFIG_FIELDS.items():
        value = getattr(args, arg_name)
        if value is not None:
            overrides[field_name] = value
    if args.no_nesterov:
        overrides["nesterov"] = False
    return overrides


def _resolve_experiment_name(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    """Return (experiment_name, yaml_data).

    yaml_data is {} when --config is not given.
    """
    if args.config is None:
        return args.experiment or DEFAULT_EXPERIMENT, {}

    yaml_data = load_yaml(Path(args.config))
    yaml_experiment = yaml_data["experiment"]

    if args.experiment is not None and args.experiment != yaml_experiment:
        raise ValueError(
            f"--experiment ({args.experiment}) conflicts with YAML 'experiment: {yaml_experiment}'"
        )
    if yaml_experiment not in EXPERIMENTS:
        raise ValueError(f"YAML 'experiment: {yaml_experiment}' is not registered")

    return yaml_experiment, yaml_data


def build_run(args: argparse.Namespace) -> RunParams:
    """Merge CLI + YAML + registry defaults into run parameters.

    Priority (low → high):
        TrainConfig default < Experiment.config < YAML < CLI < stage.overrides

    Returns (experiment_name, config, seed, amp, num_train, stages).
    """
    experiment_name, yaml_data = _resolve_experiment_name(args)
    experiment = EXPERIMENTS[experiment_name]

    # Start from the experiment's default config
    config = experiment.config

    # YAML overrides
    if yaml_data:
        yaml_over = split_overrides(yaml_data)
        validate_override_keys(yaml_over)
        yaml_over = coerce_overrides(yaml_over)
        config = dataclasses.replace(config, **yaml_over)

    # CLI overrides (highest priority at the top level)
    config = dataclasses.replace(config, **_cli_overrides(args))

    # Stages
    stages: list[Stage] | None = None
    if "stages" in yaml_data:
        stages = parse_stages(yaml_data["stages"])

    # Global flags
    seed = args.seed if args.seed is not None else experiment.seed
    amp = args.amp or experiment.amp

    return RunParams(
        experiment_name=experiment_name,
        config=config,
        seed=seed,
        amp=amp,
        num_train=args.num_train,
        stages=stages,
    )
