"""Tests for RunArtifacts."""

from pathlib import Path

from experiment.artifacts import RunArtifacts
from experiment.spec import TrainConfig
from training.strategy import SingleDeviceStrategy

_TEST_DATASET = "cifar10"


class _NonMainStrategy(SingleDeviceStrategy):
    def is_main_process(self) -> bool:
        return False


def test_create_returns_correct_paths(tmp_path: Path, any_experiment: str) -> None:
    artifacts = RunArtifacts.create(
        experiment_name=any_experiment,
        dataset_name=_TEST_DATASET,
        config=TrainConfig(),
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    assert artifacts.is_main is True
    assert artifacts.logger is not None

    assert artifacts.base.startswith(any_experiment)
    assert artifacts.log_path.name == f"{artifacts.base}.log"
    assert artifacts.ckpt_path.name == f"{artifacts.base}.pt"
    assert artifacts.cfg_path.name == f"{artifacts.base}.json"
    assert artifacts.csv_path.name == f"{artifacts.base}.csv"

    artifacts.close()


def test_create_writes_config_snapshot(tmp_path: Path, any_experiment: str) -> None:
    artifacts = RunArtifacts.create(
        experiment_name=any_experiment,
        dataset_name=_TEST_DATASET,
        config=TrainConfig(batch_size=64),
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    artifacts.close()

    assert artifacts.cfg_path.exists()
    content = artifacts.cfg_path.read_text()
    assert f'"experiment": "{any_experiment}"' in content
    assert f'"dataset": "{_TEST_DATASET}"' in content
    assert '"batch_size": 64' in content


def test_non_main_process_skips_logger_and_writer(tmp_path: Path, any_experiment: str) -> None:
    artifacts = RunArtifacts.create(
        experiment_name=any_experiment,
        dataset_name=_TEST_DATASET,
        config=TrainConfig(),
        strategy=_NonMainStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    assert artifacts.is_main is False
    assert artifacts.logger is not None
    assert not artifacts.cfg_path.exists()


def test_resume_reuses_old_csv(tmp_path: Path, any_experiment: str) -> None:
    resume_path = str(tmp_path / "checkpoints" / f"{any_experiment}_20260921_120000.pt")

    artifacts = RunArtifacts.create(
        experiment_name=any_experiment,
        dataset_name=_TEST_DATASET,
        config=TrainConfig(),
        strategy=SingleDeviceStrategy(),
        resume_path=resume_path,
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    artifacts.close()

    assert artifacts.csv_path.name == f"{any_experiment}_20260921_120000.csv"


def test_fresh_run_uses_new_csv(tmp_path: Path, any_experiment: str) -> None:
    artifacts = RunArtifacts.create(
        experiment_name=any_experiment,
        dataset_name=_TEST_DATASET,
        config=TrainConfig(),
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    artifacts.close()

    assert artifacts.csv_path.name == f"{artifacts.base}.csv"


def test_close_is_idempotent(tmp_path: Path, any_experiment: str) -> None:
    artifacts = RunArtifacts.create(
        experiment_name=any_experiment,
        dataset_name=_TEST_DATASET,
        config=TrainConfig(),
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    artifacts.close()
