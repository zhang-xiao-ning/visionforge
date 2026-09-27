"""Tests for training strategies."""

import torch
from torch.utils.data import TensorDataset

from training.strategy import (
    SingleDeviceStrategy,
    build_strategy,
)


def _make_dataset() -> TensorDataset:
    return TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))


def test_single_device_is_main_process() -> None:
    strategy = SingleDeviceStrategy()
    assert strategy.is_main_process() is True


def test_single_device_wrap_model_returns_same_model() -> None:
    strategy = SingleDeviceStrategy()
    model = torch.nn.Linear(4, 2)
    wrapped = strategy.wrap_model(model, torch.device("cpu"))
    assert wrapped is model


def test_single_device_train_sampler_covers_dataset() -> None:
    strategy = SingleDeviceStrategy()
    dataset = _make_dataset()
    sampler = strategy.make_train_sampler(dataset)
    assert sorted(sampler) == list(range(16))


def test_single_device_val_sampler_covers_dataset() -> None:
    strategy = SingleDeviceStrategy()
    dataset = _make_dataset()
    sampler = strategy.make_val_sampler(dataset)
    assert sorted(sampler) == list(range(16))


def test_single_device_set_epoch_is_noop() -> None:
    strategy = SingleDeviceStrategy()
    strategy.set_epoch(5)  # should not raise


def test_single_device_cleanup_is_noop() -> None:
    strategy = SingleDeviceStrategy()
    strategy.cleanup()  # should not raise


def test_build_strategy_returns_single_device_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    strategy = build_strategy(torch.device("cpu"))
    assert isinstance(strategy, SingleDeviceStrategy)
