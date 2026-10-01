"""Entry point for training."""

from cli import build_config, parse_args
from experiment.runner import ExperimentRunner
from framework import set_seed


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    runner = ExperimentRunner(
        cfg=cfg,
        batch_size=args.batch_size,
        resume_path=args.resume,
        num_train=args.num_train,
    )
    try:
        runner.run()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
