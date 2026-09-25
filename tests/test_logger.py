from utils.logger import CSVRecorder


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
