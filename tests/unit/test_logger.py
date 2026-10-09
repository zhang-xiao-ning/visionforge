"""Tests for AppLogger."""

import logging
from pathlib import Path

import pytest

from utils.logger import AppLogger, parse_level


def _make_logger(tmp_path: Path, **kwargs) -> AppLogger:
    """AppLogger with CSV channel enabled by default."""
    return AppLogger(
        "test_record",
        tmp_path / "test.log",
        csv_path=tmp_path / "test.csv",
        **kwargs,
    )


def test_close_releases_file_handler(tmp_path: Path) -> None:
    log_path = tmp_path / "test.log"
    logger = AppLogger("test_close_handlers", log_path)
    assert logger.raw.handlers  # handler exists before close
    logger.close()
    assert logger.raw.handlers == []  # cleared after close


# ---------- parse_level ----------


def test_parse_level_known() -> None:
    assert parse_level("none") > logging.CRITICAL
    assert parse_level("error") == logging.ERROR
    assert parse_level("info") == logging.INFO
    assert parse_level("flow") == 21
    assert parse_level("debug") == logging.DEBUG


def test_parse_level_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        parse_level("nope")


# ---------- level control ----------


def test_default_level_is_info(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VISIONFORGE_LOG_LEVEL", raising=False)
    logger = AppLogger("test_default", tmp_path / "test.log")
    assert logger.raw.level == logging.INFO


def test_env_var_controls_level(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONFORGE_LOG_LEVEL", "debug")
    logger = AppLogger("test_env", tmp_path / "test.log")
    assert logger.raw.level == logging.DEBUG


def test_none_level_silences_errors_too(tmp_path: Path) -> None:
    log_path = tmp_path / "test.log"
    logger = AppLogger("test_none", log_path, level="none")
    logger.error("should not appear")
    logger.info("should not appear")
    assert log_path.read_text() == ""


def test_error_level_only_logs_errors(tmp_path: Path) -> None:
    log_path = tmp_path / "test_err.log"
    logger = AppLogger("test_err", log_path, level="error")
    logger.info("info msg")
    logger.error("error msg")
    content = log_path.read_text()
    assert "info msg" not in content
    assert "error msg" in content


def test_flow_level_logs_flow_and_above(tmp_path: Path) -> None:
    log_path = tmp_path / "test_flow.log"
    logger = AppLogger("test_flow", log_path, level="flow")
    logger.debug("debug msg")
    logger.info("info msg")
    logger.flow("flow msg")
    logger.error("error msg")
    content = log_path.read_text()
    assert "debug msg" not in content
    assert "info msg" not in content
    assert "flow msg" in content
    assert "error msg" in content


def test_call_with_level_kwarg(tmp_path: Path) -> None:
    log_path = tmp_path / "test_call.log"
    logger = AppLogger("test_call", log_path, level="debug")
    logger("hello", level="debug")
    assert "hello" in log_path.read_text()


# ---------- CSV channel (via record) ----------


def test_record_writes_csv_header(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    content = (tmp_path / "test.csv").read_text()
    assert "epoch,train_loss,val_metric,lr" in content
    logger.close()


def test_record_appends_row(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    logger.record(1, 1.5, {"acc": 0.8, "loss": 0.5}, "acc", 0.01)
    logger.close()

    lines = (tmp_path / "test.csv").read_text().strip().split("\n")
    assert len(lines) == 2
    assert lines[1].startswith("1,1.500000,0.800000")


def test_record_uses_primary_metric(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path)
    logger.record(1, 1.5, {"acc": 0.8, "loss": 0.5}, "loss", 0.01)
    logger.close()

    lines = (tmp_path / "test.csv").read_text().strip().split("\n")
    # Second column is val_metric — should be loss (0.5), not acc (0.8)
    assert "0.500000" in lines[1]
    assert "0.800000" not in lines[1]


def test_record_append_mode(tmp_path: Path) -> None:
    logger1 = _make_logger(tmp_path)
    logger1.record(1, 1.5, {"acc": 0.8}, "acc", 0.01)
    logger1.close()

    logger2 = _make_logger(tmp_path, csv_append=True)
    logger2.record(2, 1.2, {"acc": 0.85}, "acc", 0.005)
    logger2.close()

    lines = (tmp_path / "test.csv").read_text().strip().split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert lines[2].startswith("2,1.200000,0.850000")


def test_record_without_csv_channel(tmp_path: Path) -> None:
    """record() is safe even without a CSV path (no-op for CSV channel)."""
    logger = AppLogger("test_no_csv", tmp_path / "test.log")  # no csv_path
    logger.record(1, 1.0, {"acc": 0.5}, "acc", 0.01)  # should not raise
    assert not (tmp_path / "test.csv").exists()
    logger.close()


# ---------- TensorBoard channel ----------


def test_record_writes_tb_event(tmp_path: Path) -> None:
    tb_dir = tmp_path / "tb"
    logger = _make_logger(tmp_path, tb_dir=tb_dir)
    logger.record(1, 1.5, {"acc": 0.8}, "acc", 0.01)
    logger.close()

    # TensorBoard writes event files into tb_dir
    files = list(tb_dir.iterdir())
    assert files, "expected TensorBoard event files"
    assert any(f.name.startswith("events.out.tfevents") for f in files)


def test_close_is_idempotent(tmp_path: Path) -> None:
    logger = _make_logger(tmp_path, tb_dir=tmp_path / "tb")
    logger.record(1, 1.0, {"acc": 0.5}, "acc", 0.01)
    logger.close()
    logger.close()  # should not raise
