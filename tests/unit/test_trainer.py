"""Tests for the training loop."""

import torch
import torch.optim as optim

from application.tasks import ClassificationTask
from framework.experiment.spec import TrainConfig
from framework.training import SingleDeviceStrategy, TrainHooks, train


def _train(dummy_model, dummy_loader, dummy_logger, epochs=1, **kwargs):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    cfg = TrainConfig(epochs=epochs)
    return train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        cfg,
        SingleDeviceStrategy(),
        TrainHooks(dummy_logger),
        scheduler,
        **kwargs,
    )


def test_train_runs_one_epoch(dummy_model, dummy_loader, dummy_logger):
    result = _train(dummy_model, dummy_loader, dummy_logger, epochs=1)
    assert result["last_epoch"] == 1


def test_train_updates_weights(dummy_model, dummy_loader, dummy_logger):
    w_before = dummy_model.fc.weight.detach().clone()
    _train(dummy_model, dummy_loader, dummy_logger, epochs=1)
    w_after = dummy_model.fc.weight.detach().clone()
    assert not torch.equal(w_before, w_after)


def test_train_respects_epochs(dummy_model, dummy_loader, dummy_logger):
    result = _train(dummy_model, dummy_loader, dummy_logger, epochs=3)
    assert result["last_epoch"] == 3


def test_train_with_scheduler(dummy_model, dummy_loader, dummy_logger):
    """StepLR now steps once per optimizer step (was: once per epoch).

    4 batches/epoch, 2 epochs -> 8 optimizer steps. Starting at 0.1,
    gamma=0.5, step_size=1 -> 0.1 * 0.5**8.
    """
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
        TrainHooks(dummy_logger),
        scheduler=scheduler,
    )
    assert abs(optimizer.param_groups[0]["lr"] - 0.1 * 0.5**8) < 1e-9


def test_train_stops_when_hook_returns_false(dummy_model, dummy_loader, dummy_logger):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
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
        TrainHooks(dummy_logger, on_epoch_end=on_epoch_end),
        scheduler,
    )
    assert result["last_epoch"] == stop_after


def test_train_on_epoch_end_receives_metrics(dummy_model, dummy_loader, dummy_logger):
    """The hook gets a populated EpochContext."""
    captured = {}

    def on_epoch_end(ctx) -> bool:
        captured["epoch"] = ctx.epoch
        captured["avg_loss"] = ctx.avg_loss
        captured["lr"] = ctx.lr
        return True

    _train(dummy_model, dummy_loader, dummy_logger, epochs=2)
    # Run again with the real hook
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        TrainConfig(epochs=2),
        SingleDeviceStrategy(),
        TrainHooks(dummy_logger, on_epoch_end=on_epoch_end),
        scheduler,
    )
    assert captured["epoch"] == 2
    assert captured["avg_loss"] > 0
    assert captured["lr"] > 0


def test_train_steps_scheduler_once_per_optimizer_step(dummy_model, dummy_loader, dummy_logger):
    """Regression: scheduler.step() must fire exactly once per optimizer
    step, not additionally at epoch boundaries."""
    optimizer = optim.SGD(dummy_model.parameters(), lr=1.0)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)

    # 4 batches, accum_steps=1 → 4 optimizer steps, 1 epoch
    train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        TrainConfig(epochs=1),
        SingleDeviceStrategy(),
        TrainHooks(dummy_logger),
        scheduler,
    )
    # LambdaLR: lr = 1.0 * 0.5**step_after_last_call
    # After 4 optimizer steps, scheduler.step() should have been called 4 times.
    assert abs(optimizer.param_groups[0]["lr"] - 0.5**4) < 1e-9


def test_train_reports_mean_lr_over_epoch(dummy_model, dummy_loader, dummy_logger):
    """CSV lr column must be a mean over optimizer steps, not the
    endpoint. Regression: with a per-step schedule whose start and end
    are both 0 (single-epoch warmup + cosine), the endpoint LR is 0
    even though training happened."""
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5**step)

    captured: dict[str, float] = {}

    def on_epoch_end(ctx) -> bool:
        captured["lr"] = ctx.lr
        return True

    # 4 batches, accum_steps=1 → 4 optimizer steps.
    # LRs actually used: 0.1, 0.05, 0.025, 0.0125 → mean = 0.046875.
    train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        TrainConfig(epochs=1),
        SingleDeviceStrategy(),
        TrainHooks(dummy_logger, on_epoch_end=on_epoch_end),
        scheduler,
    )
    assert abs(captured["lr"] - 0.046875) < 1e-9
