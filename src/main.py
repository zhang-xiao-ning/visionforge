"""Entry point for training."""

from cli import build_run, parse_args
from experiment.runner import ExperimentRunner
from framework import set_seed


def main() -> None:
    args = parse_args()
    experiment_name, config, seed, amp, num_train, stages = build_run(args)
    set_seed(seed)

    runner = ExperimentRunner(
        experiment_name=experiment_name,
        config=config,
        stages=stages,
        amp=amp,
        resume_path=args.resume,
        num_train=num_train,
    )
    try:
        runner.run()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
