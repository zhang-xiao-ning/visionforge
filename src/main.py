"""Entry point for training."""

from cli import build_config, parse_args
from config import device
from framework import ExperimentRunner, build_strategy, set_seed


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    strategy = build_strategy(device)
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
