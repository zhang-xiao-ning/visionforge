"""Command-line interface: argparse → TrainConfig."""

import argparse

from experiment.config import TrainConfig
from registry import EXPERIMENTS
from runtime import DEFAULT_EXPERIMENT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument(
        "--experiment", type=str, default=DEFAULT_EXPERIMENT, choices=list(EXPERIMENTS.keys())
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no-nesterov", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lr-scheduler", type=str, default="none", choices=["none", "step", "cosine"]
    )
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Adapt CLI Namespace → TrainConfig."""
    default_lr = EXPERIMENTS[args.experiment]["lr"]
    lr = args.learning_rate if args.learning_rate is not None else default_lr
    return TrainConfig(
        experiment=args.experiment,
        epochs=args.epochs,
        learning_rate=lr,
        momentum=args.momentum,
        nesterov=not args.no_nesterov,
        seed=args.seed,
        lr_scheduler=args.lr_scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        early_stop_patience=args.early_stop_patience,
        amp=args.amp,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        accum_steps=args.accum_steps,
        grad_clip=args.grad_clip,
    )
