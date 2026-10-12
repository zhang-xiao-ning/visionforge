"""Tests for training strategies."""

import pytest
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset
from torch.utils.data.distributed import DistributedSampler

from framework.training import (
    DDPStrategy,
    SingleDeviceStrategy,
    build_strategy,
)


def _make_dataset() -> TensorDataset:
    return TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,)))


def test_single_device_is_main_process() -> None:
    strategy = SingleDeviceStrategy()
    assert strategy.is_main_process() is True


def test_single_device_wrap_model_returns_same_model() -> None:
    strategy = SingleDeviceStrategy(device=torch.device("cpu"))
    model = torch.nn.Linear(4, 2)
    wrapped = strategy.wrap_model(model)
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


def test_single_device_on_epoch_start_is_noop() -> None:
    strategy = SingleDeviceStrategy()
    strategy.on_epoch_start(5)  # should not raise


def test_single_device_cleanup_is_noop() -> None:
    strategy = SingleDeviceStrategy()
    strategy.cleanup()  # should not raise


def test_build_strategy_returns_single_device_by_default(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    strategy = build_strategy()
    assert isinstance(strategy, SingleDeviceStrategy)


# ---------- DDPStrategy ----------
#
# Cannot call DDPStrategy.__init__ in a unit test: it calls
# dist.init_process_group(). Use __new__ and set only the attributes
# the methods under test actually touch.


def _ddp_strategy() -> DDPStrategy:
    strategy = DDPStrategy.__new__(DDPStrategy)
    strategy.rank = 0
    strategy.local_rank = 0
    strategy._device = torch.device("cpu")
    strategy._train_sampler = None
    return strategy


@pytest.fixture
def fake_single_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make DistributedSampler believe we're rank 0 of a 1-process group."""
    monkeypatch.setattr(dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)


def test_ddp_make_train_sampler_caches_distributed_sampler(
    fake_single_rank: None,
) -> None:
    strategy = _ddp_strategy()
    sampler = strategy.make_train_sampler(_make_dataset())

    assert isinstance(sampler, DistributedSampler)
    assert strategy._train_sampler is sampler


def test_ddp_on_epoch_start_forwards_epoch_to_sampler(
    fake_single_rank: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy = _ddp_strategy()
    sampler = strategy.make_train_sampler(_make_dataset())

    seen: list[int] = []
    monkeypatch.setattr(sampler, "set_epoch", seen.append)

    strategy.on_epoch_start(1)
    strategy.on_epoch_start(2)

    assert seen == [1, 2]


def test_ddp_on_epoch_start_is_noop_before_sampler_exists() -> None:
    """on_epoch_start may fire before any dataset is built; must not raise."""
    _ddp_strategy().on_epoch_start(1)
