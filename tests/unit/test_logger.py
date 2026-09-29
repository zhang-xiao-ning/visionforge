import logging

import pytest

from utils.logger import AppLogger, CSVRecorder, parse_level


def test_csv_recorder_writes_header(tmp_path):
    csv_path = tmp_path / "test.csv"
    CSVRecorder(csv_path)

    content = csv_path.read_text()
    assert "epoch,train_loss,val_metric,lr" in content


def test_csv_recorder_appends_row(tmp_path):
    csv_path = tmp_path / "test.csv"
    recorder = CSVRecorder(csv_path)
    recorder.log(1, 1.5, 0.8, 0.01)

    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert lines[1].startswith("1,1.500000,0.800000")


def test_csv_recorder_append_mode(tmp_path):
    csv_path = tmp_path / "test.csv"

    r1 = CSVRecorder(csv_path)
    r1.log(1, 1.5, 0.8, 0.01)

    r2 = CSVRecorder(csv_path, append=True)
    r2.log(2, 1.2, 0.85, 0.005)

    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 3  # header + 2 rows
    assert lines[2].startswith("2,1.200000,0.850000")


def test_csv_recorder_without_lr(tmp_path):
    csv_path = tmp_path / "test.csv"
    recorder = CSVRecorder(csv_path)
    recorder.log(1, 1.5, 0.8)

    lines = csv_path.read_text().strip().split("\n")
    assert lines[1].endswith(",")  # lr 为空


def test_parse_level_known() -> None:
    assert parse_level("none") > logging.CRITICAL
    assert parse_level("error") == logging.ERROR
    assert parse_level("info") == logging.INFO
    assert parse_level("flow") == 21
    assert parse_level("debug") == logging.DEBUG


def test_parse_level_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown log level"):
        parse_level("nope")


def test_default_level_is_info(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VISIONFORGE_LOG_LEVEL", raising=False)
    logger = AppLogger("test_default", tmp_path / "test.log")
    assert logger.raw.level == logging.INFO


def test_env_var_controls_level(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VISIONFORGE_LOG_LEVEL", "debug")
    logger = AppLogger("test_env", tmp_path / "test.log")
    assert logger.raw.level == logging.DEBUG


def test_none_level_silences_everything(tmp_path, capsys) -> None:
    logger = AppLogger("test_none", tmp_path / "test.log", level="none")
    logger.error("should not appear")
    logger.info("should not appear")
    captured = capsys.readouterr()
    assert captured.out == ""


def test_flow_level_above_info(tmp_path, capsys) -> None:
    logger = AppLogger("test_flow", tmp_path / "test.log", level="info")
    logger.flow("epoch start")
    captured = capsys.readouterr()
    assert "FLOW" in captured.err or "epoch start" in captured.err


def test_call_with_level_kwarg(tmp_path, capsys) -> None:
    logger = AppLogger("test_call", tmp_path / "test.log", level="debug")
    logger("hello", level="debug")
    captured = capsys.readouterr()
    assert "hello" in captured.err
