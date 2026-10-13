"""Tests for dataset registry."""

import pytest

from application.datasets.cifar10 import VAL_SIZE, _train_val_indices
from framework.registry import DATASETS
from framework.training import SingleDeviceStrategy


def test_registry_has_cifar10() -> None:
    assert "cifar10" in DATASETS


def test_registry_has_flickr8k() -> None:
    assert "flickr8k" in DATASETS


def test_unknown_dataset_raises() -> None:
    from framework.interfaces.bundle import DataContext

    with pytest.raises(ValueError, match="Unknown dataset"):
        DATASETS.build("nope", DataContext(batch_size=4, strategy=SingleDeviceStrategy()))


def test_train_val_default_split() -> None:
    train, val = _train_val_indices(50_000, None)
    assert len(train) == 50_000 - VAL_SIZE
    assert len(val) == VAL_SIZE
    assert train.stop <= val.start  # no overlap


def test_train_val_num_train_caps_only_train() -> None:
    """--num-train must shrink train, NOT grow val."""
    train, val = _train_val_indices(50_000, 1000)
    assert len(train) == 1000
    assert len(val) == VAL_SIZE  # ← the invariant that was broken


def test_train_val_num_train_above_pool_is_capped() -> None:
    train, val = _train_val_indices(50_000, 49_999)
    assert len(train) == 50_000 - VAL_SIZE  # capped
    assert len(val) == VAL_SIZE


def test_train_val_no_overlap() -> None:
    for n in (None, 0, 100, 49_000, 49_999):
        train, val = _train_val_indices(50_000, n)
        assert train.stop <= val.start
