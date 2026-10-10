"""Tests for corpus BLEU-4."""

import pytest

from framework.metrics import corpus_bleu


def test_perfect_match() -> None:
    score = corpus_bleu(["a cat on the mat"], [["a cat on the mat"]])
    assert score == pytest.approx(1.0)


def test_no_match() -> None:
    score = corpus_bleu(["a"], [["z"]])
    assert score == 0.0


def test_partial_match() -> None:
    score = corpus_bleu(["a cat is on the mat"], [["the cat is on the mat"]])
    assert 0.0 < score < 1.0


def test_multiple_references() -> None:
    score = corpus_bleu(
        ["a cat on the mat"],
        [["a cat on the mat", "the cat on a mat", "cat on mat"]],
    )
    assert score == pytest.approx(1.0)


def test_case_insensitive() -> None:
    score = corpus_bleu(["A Cat On The Mat"], [["a cat on the mat"]])
    assert score == pytest.approx(1.0)


def test_brevity_penalty() -> None:
    long_ref = "a cat is sitting on the mat in the room"
    short_pred = "a cat on the mat"
    long_score = corpus_bleu([long_ref], [[long_ref]])
    short_score = corpus_bleu([short_pred], [[long_ref]])
    assert short_score < long_score


def test_empty_references_returns_zero() -> None:
    score = corpus_bleu(["a cat"], [[]])
    assert score == 0.0


def test_corpus_level_not_average_of_sentences() -> None:
    """Corpus BLEU uses global n-gram counts, not per-sentence averages.

    Data chosen so that per-sentence averaging would give 0.5, but
    corpus-level computation gives ~0.69.
    """
    preds = ["a b c d", "x y z"]
    refs = [["a b c d"], ["a b c"]]
    score = corpus_bleu(preds, refs)
    # Per-sentence average would be (1.0 + 0.0) / 2 = 0.5
    assert score > 0.6
