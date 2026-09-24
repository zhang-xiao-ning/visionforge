"""Tests for RunArtifacts."""

from pathlib import Path

import pytest

from experiment.artifacts import RunArtifacts
from experiment.config import TrainConfig
from training.strategy import SingleDeviceStrategy


class _NonMainStrategy(SingleDeviceStrategy):
    """Pretend to be a non-main DDP process."""

    def is_main_process(self) -> bool:
        return False


@pytest.fixture
def cfg() -> TrainConfig:
    return TrainConfig(experiment="mlp", epochs=1)


def test_create_returns_correct_paths(tmp_path: Path, cfg: TrainConfig) -> None:
    artifacts = RunArtifacts.create(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=64,
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    assert artifacts.is_main is True
    assert artifacts.logger is not None
    assert artifacts.recorder is not None
    assert artifacts.writer is not None

    assert artifacts.base.startswith("mlp_")
    assert artifacts.log_path.name == f"{artifacts.base}.log"
    assert artifacts.ckpt_path.name == f"{artifacts.base}.pt"
    assert artifacts.cfg_path.name == f"{artifacts.base}.json"
    assert artifacts.csv_path.name == f"{artifacts.base}.csv"

    artifacts.close()


def test_create_writes_config_snapshot(tmp_path: Path, cfg: TrainConfig) -> None:
    artifacts = RunArtifacts.create(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=64,
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    artifacts.close()

    assert artifacts.cfg_path.exists()
    content = artifacts.cfg_path.read_text()
    assert '"experiment": "mlp"' in content
    assert '"dataset": "cifar10"' in content
    assert '"batch_size": 64' in content


def test_non_main_process_skips_logger_and_writer(tmp_path: Path, cfg: TrainConfig) -> None:
    artifacts = RunArtifacts.create(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=64,
        strategy=_NonMainStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )

    assert artifacts.is_main is False
    assert artifacts.logger is None
    assert artifacts.recorder is None
    assert artifacts.writer is None

    # 非主进程不写任何文件
    assert not artifacts.cfg_path.exists()


def test_resume_reuses_old_csv(tmp_path: Path, cfg: TrainConfig) -> None:
    resume_path = str(tmp_path / "checkpoints" / "mlp_20260921_120000.pt")

    artifacts = RunArtifacts.create(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=64,
        strategy=SingleDeviceStrategy(),
        resume_path=resume_path,
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    artifacts.close()

    assert artifacts.csv_path.name == "mlp_20260921_120000.csv"


def test_fresh_run_uses_new_csv(tmp_path: Path, cfg: TrainConfig) -> None:
    artifacts = RunArtifacts.create(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=64,
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    artifacts.close()

    assert artifacts.csv_path.name == f"{artifacts.base}.csv"


def test_close_is_idempotent(tmp_path: Path, cfg: TrainConfig) -> None:
    artifacts = RunArtifacts.create(
        cfg=cfg,
        dataset_name="cifar10",
        batch_size=64,
        strategy=SingleDeviceStrategy(),
        outputs_dir=tmp_path / "outputs",
        checkpoints_dir=tmp_path / "checkpoints",
    )
    artifacts.close()
    artifacts.close()  # should not raise
