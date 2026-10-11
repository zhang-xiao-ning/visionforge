"""Tests for dataset registry."""

import pytest

import application.registry  # noqa: F401 — populates DATASETS
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
