"""Evaluation: metrics that know how to score a model on a loader."""

from evaluation.base import Metric
from evaluation.metrics import Accuracy, CrossEntropy, Perplexity

__all__ = ["Metric", "Accuracy", "CrossEntropy", "Perplexity"]
