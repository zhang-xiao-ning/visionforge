"""Evaluation: metrics that know how to score a model on a loader."""

from evaluation.base import Metric
from evaluation.bleu import BLEU4, corpus_bleu
from evaluation.metrics import Accuracy, CrossEntropy, Perplexity

__all__ = ["Metric", "Accuracy", "CrossEntropy", "Perplexity", "BLEU4", "corpus_bleu"]
