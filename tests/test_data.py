import pytest

from data.datasets import DATASET_REGISTRY


def test_registry_has_cifar10():
    assert "cifar10" in DATASET_REGISTRY


def test_unknown_dataset_raises():
    from data.datasets import build_loaders
    with pytest.raises(ValueError):
        build_loaders(name="no_such_dataset")


@pytest.mark.skip(reason="needs real data on disk")
def test_build_loaders_real():
    from data.datasets import build_loaders
    loader_train, loader_val, loader_test = build_loaders(
        name="cifar10", batch_size=64,
    )
    x, y = next(iter(loader_train))
    assert x.shape == (64, 3, 32, 32)
    assert y.shape == (64,)