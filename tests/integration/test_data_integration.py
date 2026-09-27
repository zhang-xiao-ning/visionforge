"""Integration tests for data loading with real CIFAR-10 data."""

import pytest

from data.datasets import build_loaders

pytestmark = pytest.mark.integration


def test_build_loaders_real(require_integration, require_cifar10) -> None:
    """Full pipeline: dataset files → DataLoader."""
    loader_train, loader_val, loader_test = build_loaders(name="cifar10", batch_size=64)
    x, y = next(iter(loader_train))
    assert x.shape == (64, 3, 32, 32)
    assert y.shape == (64,)
