"""Tests for dataset registry."""

import pytest

from data.datasets import DATASET_REGISTRY, build_data
from training.strategy import SingleDeviceStrategy


def test_registry_has_cifar10() -> None:
    assert "cifar10" in DATASET_REGISTRY


def test_registry_has_flickr8k() -> None:
    assert "flickr8k" in DATASET_REGISTRY


def test_unknown_dataset_raises() -> None:
    from framework.interfaces.bundle import DataContext

    with pytest.raises(ValueError, match="Unknown dataset"):
        build_data("nope", DataContext(batch_size=4, strategy=SingleDeviceStrategy()))
