import torch
import torch.optim as optim

from tasks.classification import ClassificationTask
from training.train import train


def test_train_runs_one_epoch(dummy_model, dummy_loader):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    result = train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        epochs=1,
    )
    assert "best_acc" in result
    assert "last_epoch" in result
    assert result["last_epoch"] == 1
    assert 0.0 <= result["best_acc"] <= 1.0


def test_train_updates_weights(dummy_model, dummy_loader):
    w_before = dummy_model.fc.weight.detach().clone()

    optimizer = optim.SGD(dummy_model.parameters(), lr=0.1)
    train(dummy_model, optimizer, dummy_loader, dummy_loader, ClassificationTask(), epochs=1)

    w_after = dummy_model.fc.weight.detach().clone()
    assert not torch.equal(w_before, w_after)


def test_train_respects_epochs(dummy_model, dummy_loader):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    result = train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        epochs=3,
    )
    assert result["last_epoch"] == 3


def test_train_with_scheduler(dummy_model, dummy_loader):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        epochs=2,
        scheduler=scheduler,
    )

    # 2 epoch 后 lr 应该衰减两次：0.1 -> 0.05 -> 0.025
    assert abs(optimizer.param_groups[0]["lr"] - 0.025) < 1e-9


def test_train_early_stopping(dummy_model, dummy_loader):
    optimizer = optim.SGD(dummy_model.parameters(), lr=0.01)
    result = train(
        dummy_model,
        optimizer,
        dummy_loader,
        dummy_loader,
        ClassificationTask(),
        epochs=100,
        early_stop_patience=2,
    )
    # 数据是随机的，验证集不会持续提升，应该触发早停
    assert result["last_epoch"] < 100
