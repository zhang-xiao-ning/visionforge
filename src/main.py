"""Entry point for training."""

from cli import build_run, parse_args
from experiment.runner import ExperimentRunner
from framework import set_seed


def main() -> None:
    args = parse_args()
    params = build_run(args)
    set_seed(params.seed)

    runner = ExperimentRunner(
        experiment_name=params.experiment_name,
        config=params.config,
        stages=params.stages,
        amp=params.amp,
        resume_path=args.resume,
        num_train=params.num_train,
    )
    try:
        runner.run()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
