"""Integration tests for data loading with real CIFAR-10 data."""

import pytest

from data.datasets import build_data
from framework.interfaces.bundle import DataContext
from training.strategy import SingleDeviceStrategy

pytestmark = pytest.mark.integration


def test_build_data_real(require_integration, require_cifar10) -> None:
    """Full pipeline: dataset files → DataBundle."""
    bundle, eval_bundle = build_data(
        "cifar10",
        DataContext(batch_size=64, strategy=SingleDeviceStrategy()),
    )
    x, y = next(iter(bundle.loader_train))
    assert x.shape == (64, 3, 32, 32)
    assert y.shape == (64,)
    assert bundle.model_init == {"num_classes": 10}
    assert eval_bundle is None  # cifar10 needs no separate eval set
