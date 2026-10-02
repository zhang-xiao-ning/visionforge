"""Tests for built-in metrics."""

import torch
import torch.nn as nn

from evaluation import Accuracy, CrossEntropy, Perplexity
from evaluation.base import Metric


class _DummyClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


def test_accuracy_implements_protocol() -> None:
    assert isinstance(Accuracy(), Metric)


def test_cross_entropy_implements_protocol() -> None:
    assert isinstance(CrossEntropy(), Metric)


def test_perplexity_implements_protocol() -> None:
    assert isinstance(Perplexity(), Metric)


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


def test_accuracy_restores_train_mode(dummy_loader) -> None:
    model = _DummyClassifier()
    model.eval()
    Accuracy().evaluate(model, dummy_loader, torch.device("cpu"), torch.float32)
    assert model.training is True


def test_metric_is_stateless(dummy_loader) -> None:
    """Calling twice on the same loader gives the same result."""
    model = _DummyClassifier()
    m = Accuracy()
    a = m.evaluate(model, dummy_loader, torch.device("cpu"), torch.float32)
    b = m.evaluate(model, dummy_loader, torch.device("cpu"), torch.float32)
    assert a == b
