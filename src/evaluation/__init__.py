"""Evaluation: metrics that know how to score a model on a loader."""

from evaluation.bleu import BLEU4, corpus_bleu
from evaluation.metrics import Accuracy, CrossEntropy, Perplexity
from framework.interfaces import Matric

__all__ = ["Matric", "Accuracy", "CrossEntropy", "Perplexity", "BLEU4", "corpus_bleu"]
