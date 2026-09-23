"""Entry point for training."""

import argparse

from config import TrainConfig
from framework import ExperimentRunner, build_strategy, set_seed
from registry import EXPERIMENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument("--experiment", type=str, default="mlp", choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10")
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

    parser.add_argument(
        "--resume", type=str, default=None, help="path to checkpoint to resume from"
    )
    parser.add_argument("--amp", action="store_true", help="enable mixed precision (CUDA only)")

    parser.add_argument(
        "--num-train", type=int, default=None, help="number of training samples (default: all)"
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    _, default_lr = EXPERIMENTS[args.experiment]
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
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    strategy = build_strategy()
    runner = ExperimentRunner(
        cfg=cfg,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        strategy=strategy,
        resume_path=args.resume,
        num_train=args.num_train,
    )
    try:
        runner.run()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
