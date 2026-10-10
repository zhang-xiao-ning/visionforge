"""Entry point for training."""

from cli import build_run, parse_args
from framework.experiment.runner import ExperimentRunner
from framework.registry import EXPERIMENTS
from framework.utils import set_seed


def main() -> None:
    args = parse_args()
    params = build_run(args)
    set_seed(params.seed)
    experiment = EXPERIMENTS.get(params.experiment_name)
    runner = ExperimentRunner(
        experiment=experiment,
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
