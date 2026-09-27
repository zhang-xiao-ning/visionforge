import pytest

from data.datasets import DATASET_REGISTRY


def test_registry_has_cifar10():
    assert "cifar10" in DATASET_REGISTRY


def test_unknown_dataset_raises():
    from data.datasets import build_loaders

    with pytest.raises(ValueError):
        build_loaders(name="no_such_dataset")
