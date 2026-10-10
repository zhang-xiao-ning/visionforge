"""Tests for built-in metrics."""

import pytest
import torch
import torch.nn as nn

from evaluation import Accuracy, CrossEntropy, Perplexity
from framework.interfaces import Matric, eval_mode


class _DummyClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class _FailingMetric:
    """Metric whose evaluate always raises."""


def test_eval_mode_restores_train_on_exception() -> None:
    """If the body raises, model must be back in train mode."""
    model = _DummyClassifier()
    model.train()

    with pytest.raises(RuntimeError):
        with eval_mode(model):
            raise RuntimeError("boom")

    assert model.training is True


def test_eval_mode_preserves_eval_if_was_eval() -> None:
    """If model was in eval before, stay in eval after."""
    model = _DummyClassifier()
    model.eval()

    with eval_mode(model):
        pass

    assert model.training is False


def test_metric_does_not_force_train_mode(dummy_loader) -> None:
    """If a caller invokes evaluate on an eval-mode model, it stays eval."""
    model = _DummyClassifier()
    model.eval()
    Accuracy().evaluate(model, dummy_loader, torch.device("cpu"), torch.float32)
    assert model.training is False


def test_accuracy_implements_protocol() -> None:
    assert isinstance(Accuracy(), Matric)


def test_cross_entropy_implements_protocol() -> None:
    assert isinstance(CrossEntropy(), Matric)


def test_perplexity_implements_protocol() -> None:
    assert isinstance(Perplexity(), Matric)


def test_metric_names_and_directions() -> None:
    assert Accuracy.name == "acc"
    assert Accuracy.higher_is_better is True
    assert CrossEntropy.name == "loss"
    assert CrossEntropy.higher_is_better is False
    assert Perplexity.name == "perplexity"
    assert Perplexity.higher_is_better is False


def test_accuracy_range(dummy_loader) -> None:
    model = _DummyClassifier()
    acc = Accuracy().evaluate(model, dummy_loader, torch.device("cpu"), torch.float32)
    assert 0.0 <= acc <= 1.0


def test_cross_entropy_positive(dummy_loader) -> None:
    model = _DummyClassifier()
    loss = CrossEntropy().evaluate(model, dummy_loader, torch.device("cpu"), torch.float32)
    assert loss > 0


def test_metric_is_stateless(dummy_loader) -> None:
    """Calling twice on the same loader gives the same result."""
    model = _DummyClassifier()
    m = Accuracy()
    a = m.evaluate(model, dummy_loader, torch.device("cpu"), torch.float32)
    b = m.evaluate(model, dummy_loader, torch.device("cpu"), torch.float32)
    assert a == b


def test_metrics_run_every_epoch_by_default() -> None:
    for cls in (Accuracy, CrossEntropy, Perplexity):
        assert cls.run_every_n_epochs == 1
