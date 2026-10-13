"""Tests for the tiny shakespeare dataset."""

from pathlib import Path

import pytest
import torch

from application.datasets import tinyshakespeare
from application.datasets.tinyshakespeare import ShakespeareDataset, build_bundle
from framework.interfaces.bundle import DataContext
from framework.training import SingleDeviceStrategy


@pytest.fixture
def fake_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "input.txt"
    p.write_text("abcdefghij" * 500)  # 5000 chars, 10 unique
    monkeypatch.setattr(tinyshakespeare, "DEFAULT_PATH", p)
    return p


def test_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tinyshakespeare, "DEFAULT_PATH", tmp_path / "nope.txt")
    with pytest.raises(FileNotFoundError, match="tiny shakespeare"):
        build_bundle(DataContext(batch_size=4, strategy=SingleDeviceStrategy()))


def test_bundle_shapes(fake_text: Path) -> None:
    bundle, eval_bundle = build_bundle(DataContext(batch_size=4, strategy=SingleDeviceStrategy()))
    assert eval_bundle is None
    x, y = next(iter(bundle.loader_train))
    assert x.shape == (4, tinyshakespeare.BLOCK_SIZE)
    assert y.shape == x.shape
    assert x.dtype == torch.long


def test_bundle_vocab_keeps_case(fake_text: Path) -> None:
    bundle, _ = build_bundle(DataContext(batch_size=4, strategy=SingleDeviceStrategy()))
    # 10 chars + 4 specials
    assert bundle.model_init["vocab_size"] == 14
    assert bundle.model_init["block_size"] == tinyshakespeare.BLOCK_SIZE
    assert bundle.extras["lower"] is False
    assert bundle.extras["itos"][:4] == ["<pad>", "<bos>", "<eos>", "<unk>"]


def test_dataset_shifts_targets() -> None:
    ids = torch.arange(20, dtype=torch.long)
    ds = ShakespeareDataset(ids, block_size=4)
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2, 3]
    assert y.tolist() == [1, 2, 3, 4]


def test_dataset_too_short_raises() -> None:
    with pytest.raises(ValueError, match="block_size"):
        ShakespeareDataset(torch.arange(3), block_size=4)
