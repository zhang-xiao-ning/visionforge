"""Tests for the training loop."""

import torch
import torch.optim as optim

from experiment.spec import TrainConfig
from tasks.classification import ClassificationTask
from training.hooks import TrainHooks
from training.strategy import SingleDeviceStrategy
from training.train import train


def _train(dummy_model, dummy_loader, epochs=1, **kwargs):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    cfg = TrainConfig(epochs=epochs)
    return train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        cfg,
        SingleDeviceStrategy(),
        TrainHooks(),
        **kwargs,
    )


def test_train_runs_one_epoch(dummy_model, dummy_loader):
    result = _train(dummy_model, dummy_loader, epochs=1)
    assert result["last_epoch"] == 1


def test_train_updates_weights(dummy_model, dummy_loader):
    w_before = dummy_model.fc.weight.detach().clone()
    _train(dummy_model, dummy_loader, epochs=1)
    w_after = dummy_model.fc.weight.detach().clone()
    assert not torch.equal(w_before, w_after)


def test_train_respects_epochs(dummy_model, dummy_loader):
    result = _train(dummy_model, dummy_loader, epochs=3)
    assert result["last_epoch"] == 3


def test_train_with_scheduler(dummy_model, dummy_loader):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    cfg = TrainConfig(epochs=2)
    train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        cfg,
        SingleDeviceStrategy(),
        TrainHooks(),
        scheduler=scheduler,
    )
    assert abs(optimizer.param_groups[0]["lr"] - 0.025) < 1e-9


def test_train_stops_when_hook_returns_false(dummy_model, dummy_loader):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    cfg = TrainConfig(epochs=100)
    stop_after = 2
    count = {"n": 0}

    def on_epoch_end(ctx) -> bool:
        count["n"] += 1
        return count["n"] < stop_after

    result = train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        cfg,
        SingleDeviceStrategy(),
        TrainHooks(on_epoch_end=on_epoch_end),
    )
    assert result["last_epoch"] == stop_after


def test_train_on_epoch_end_receives_metrics(dummy_model, dummy_loader):
    """The hook gets a populated EpochContext."""
    captured = {}

    def on_epoch_end(ctx) -> bool:
        captured["epoch"] = ctx.epoch
        captured["avg_loss"] = ctx.avg_loss
        captured["lr"] = ctx.lr
        return True

    _train(dummy_model, dummy_loader, epochs=2)
    # Run again with the real hook
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        TrainConfig(epochs=2),
        SingleDeviceStrategy(),
        TrainHooks(on_epoch_end=on_epoch_end),
    )
    assert captured["epoch"] == 2
    assert captured["avg_loss"] > 0
    assert captured["lr"] > 0
