"""Entry point for training."""

import application.registry  # noqa: F401 — triggers dataset/experiment registration
from framework.cli import build_run
from framework.experiment import ExperimentRunner
from framework.registry import EXPERIMENTS
from framework.utils import set_seed


def main() -> None:
    params = build_run()
    set_seed(params.seed)
    experiment = EXPERIMENTS.get(params.experiment_name)
    runner = ExperimentRunner(
        experiment=experiment,
        config=params.config,
        stages=params.stages,
        amp=params.amp,
        resume_path=params.resume,
        num_train=params.num_train,
    )
    try:
        runner.run()
    finally:
        runner.cleanup()


if __name__ == "__main__":
    main()
