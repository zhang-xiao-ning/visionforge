"""Tests for the StudentSimilarity metric."""

import torch
import torch.nn as nn

from application.metrics.dino import StudentSimilarity
from framework.interfaces import Metric


class _FakeModel(nn.Module):
    """Maps input to L2-normalised K-dim output. Deterministic."""

    def __init__(self, K: int = 8, mode: str = "diverse") -> None:
        super().__init__()
        self.K = K
        self.mode = mode
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        if self.mode == "diverse":
            out = torch.zeros(B, self.K)
            # Each row gets a distinct one-hot direction → orthogonal.
            for i in range(B):
                out[i, i % self.K] = 1.0
        elif self.mode == "collapsed":
            out = torch.ones(B, self.K) / (self.K**0.5)
        else:  # pragma: no cover
            raise ValueError(self.mode)
        return out + self._dummy * 0.0


def _loader(n: int = 4, batch: int = 4) -> list:
    # The metric only needs batches iterable; use a plain list of
    # (view1, view2) tuples to avoid DataLoader overhead.
    x = torch.randn(n, 3, 32, 32)
    return [(x, x)]


def test_metric_implements_protocol() -> None:
    assert isinstance(StudentSimilarity(), Metric)


def test_metadata() -> None:
    assert StudentSimilarity.name == "student_sim"
    assert StudentSimilarity.higher_is_better is False
    assert StudentSimilarity.run_every_n_epochs == 1


def test_diverse_output_gives_low_similarity() -> None:
    metric = StudentSimilarity()
    value = metric.evaluate(
        _FakeModel(K=8, mode="diverse"), _loader(4, 4), torch.device("cpu"), torch.float32
    )
    assert value < 0.01


def test_collapsed_output_gives_high_similarity() -> None:
    metric = StudentSimilarity()
    value = metric.evaluate(
        _FakeModel(K=8, mode="collapsed"), _loader(4, 4), torch.device("cpu"), torch.float32
    )
    assert value > 0.99


def test_single_element_batch_is_skipped() -> None:
    metric = StudentSimilarity()
    value = metric.evaluate(
        _FakeModel(K=8),
        [(torch.randn(1, 3, 32, 32), torch.randn(1, 3, 32, 32))],
        torch.device("cpu"),
        torch.float32,
    )
    # No pairs available → default 0.0.
    assert value == 0.0
