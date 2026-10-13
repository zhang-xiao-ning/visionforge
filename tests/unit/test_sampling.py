"""Tests for framework.inference.sampling.

Design is tested, not the implementation: strategies as pure functions,
config validation, loop contract (call counts, EOS semantics), and
determinism under a seeded generator.
"""

from dataclasses import FrozenInstanceError

import pytest
import torch

from framework.inference import SamplingConfig, apply_sampling, sample

# ---------- SamplingConfig ----------


def test_default_config_is_greedy() -> None:
    cfg = SamplingConfig()
    assert cfg.strategy == "greedy"
    assert cfg.temperature == 1.0
    assert cfg.top_k is None
    assert cfg.top_p is None


def test_config_is_frozen() -> None:
    cfg = SamplingConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.strategy = "top_k"  # type: ignore[misc]


def test_greedy_rejects_top_k() -> None:
    with pytest.raises(ValueError, match="greedy"):
        SamplingConfig(strategy="greedy", top_k=50)


def test_greedy_rejects_top_p() -> None:
    with pytest.raises(ValueError, match="greedy"):
        SamplingConfig(strategy="greedy", top_p=0.9)


def test_greedy_rejects_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingConfig(strategy="greedy", temperature=0.7)


def test_top_k_requires_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SamplingConfig(strategy="top_k")


def test_top_k_rejects_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingConfig(strategy="top_k", top_k=5, top_p=0.9)


def test_top_k_rejects_nonpositive_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SamplingConfig(strategy="top_k", top_k=0)


def test_top_p_requires_top_p() -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingConfig(strategy="top_p")


def test_top_p_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="top_p"):
        SamplingConfig(strategy="top_p", top_p=1.5)
    with pytest.raises(ValueError, match="top_p"):
        SamplingConfig(strategy="top_p", top_p=0.0)


def test_top_p_rejects_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SamplingConfig(strategy="top_p", top_p=0.9, top_k=5)


def test_temperature_must_be_positive() -> None:
    with pytest.raises(ValueError, match="temperature"):
        SamplingConfig(strategy="top_k", top_k=5, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        SamplingConfig(strategy="top_k", top_k=5, temperature=-1.0)


# ---------- apply_sampling ----------


def test_apply_greedy_is_argmax() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]])
    out = apply_sampling(logits, SamplingConfig())
    assert out.tolist() == [1, 0]


def test_apply_returns_long() -> None:
    logits = torch.randn(2, 5)
    out = apply_sampling(logits, SamplingConfig())
    assert out.dtype == torch.long


def test_top_k_1_equals_greedy() -> None:
    logits = torch.tensor([[1.0, 3.0, 2.0], [0.0, -1.0, 5.0]])
    greedy = apply_sampling(logits, SamplingConfig())
    topk = apply_sampling(logits, SamplingConfig(strategy="top_k", top_k=1))
    assert torch.equal(greedy, topk)


def test_top_k_stays_in_top_k_set() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])
    top3 = set(logits.topk(3, dim=-1).indices.squeeze(0).tolist())

    cfg = SamplingConfig(strategy="top_k", top_k=3)
    for _ in range(50):
        tok = apply_sampling(logits, cfg)
        assert tok.item() in top3


def test_top_p_at_one_keeps_all_tokens() -> None:
    """top_p=1.0 with only the top token nonzero must still be able to
    pick the top token (i.e. truncation does not eat everything)."""
    logits = torch.tensor([[10.0, 9.0, 8.0]])
    out = apply_sampling(logits, SamplingConfig(strategy="top_p", top_p=1.0))
    assert out.item() in {0, 1, 2}


def test_top_p_at_tiny_value_is_argmax() -> None:
    """top_p so small only the top token survives truncation."""
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    cfg = SamplingConfig(strategy="top_p", top_p=1e-6)
    for _ in range(10):
        assert apply_sampling(logits, cfg).item() == 0


def test_low_temperature_concentrates_on_argmax() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    cfg = SamplingConfig(strategy="top_k", top_k=3, temperature=1e-6)
    for _ in range(20):
        assert apply_sampling(logits, cfg).item() == 1


def test_generator_makes_sampling_reproducible() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    cfg = SamplingConfig(strategy="top_k", top_k=4)

    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)

    a = [apply_sampling(logits, cfg, generator=g1).item() for _ in range(10)]
    b = [apply_sampling(logits, cfg, generator=g2).item() for _ in range(10)]
    assert a == b


# ---------- sample ----------


def _constant_logits_fn(token: int, vocab: int = 10):
    """logits_fn that always puts `token` at the top for every row."""

    def fn(ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        logits = torch.zeros(B, L, vocab)
        logits[..., token] = 10.0
        return logits

    return fn


def test_sample_zero_new_tokens_returns_prompt_and_never_calls() -> None:
    calls = {"n": 0}

    def fn(ids: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return torch.zeros(*ids.shape, 5)

    prompt = torch.tensor([[1, 2, 3]])
    out = sample(fn, prompt, max_new_tokens=0)
    assert torch.equal(out, prompt)
    assert calls["n"] == 0


def test_sample_appends_max_new_tokens_greedy() -> None:
    prompt = torch.tensor([[1, 2]])
    out = sample(
        _constant_logits_fn(3),
        prompt,
        max_new_tokens=4,
        config=SamplingConfig(strategy="greedy"),
    )
    assert out.tolist() == [[1, 2, 3, 3, 3, 3]]


def test_sample_calls_logits_fn_once_per_token() -> None:
    calls = {"n": 0}

    def fn(ids: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        B, L = ids.shape
        out = torch.zeros(B, L, 5)
        out[..., 0] = 10.0  # never EOS
        return out

    sample(fn, torch.tensor([[1]]), max_new_tokens=7)
    assert calls["n"] == 7


def test_sample_without_eos_runs_full_length() -> None:
    prompt = torch.tensor([[1]])
    out = sample(
        _constant_logits_fn(9),
        prompt,
        max_new_tokens=5,
        eos_id=None,
    )
    assert out.shape == (1, 6)
    assert out.tolist() == [[1, 9, 9, 9, 9, 9]]


def test_sample_stops_when_every_row_finishes() -> None:
    calls = {"n": 0}

    def fn(ids: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        B, L = ids.shape
        out = torch.zeros(B, L, 5)
        out[..., 2] = 10.0  # always EOS
        return out

    out = sample(fn, torch.tensor([[1]]), max_new_tokens=10, eos_id=2)
    assert out.tolist() == [[1, 2]]
    assert calls["n"] == 1


def test_sample_eos_fills_per_row() -> None:
    """Row 0 finishes immediately; row 1 keeps generating."""

    def fn(ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        out = torch.zeros(B, L, 10)
        out[0, -1, 9] = 10.0  # row 0: EOS
        out[1, -1, 3] = 10.0  # row 1: token 3
        return out

    prompt = torch.tensor([[1], [1]])
    out = sample(fn, prompt, max_new_tokens=3, eos_id=9)
    assert out.tolist() == [[1, 9, 9, 9], [1, 3, 3, 3]]


def test_sample_returns_long() -> None:
    out = sample(
        _constant_logits_fn(1),
        torch.tensor([[1]]),
        max_new_tokens=2,
    )
    assert out.dtype == torch.long


def test_sample_generator_is_threaded_through() -> None:
    """Same generator seed → same output across two sample() calls."""

    def fn(ids: torch.Tensor) -> torch.Tensor:
        # give all tokens equal logits so sampling is stochastic
        B, L = ids.shape
        return torch.zeros(B, L, 5)

    prompt = torch.tensor([[1]])
    cfg = SamplingConfig(strategy="top_k", top_k=5)

    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)

    a = sample(fn, prompt, max_new_tokens=5, config=cfg, generator=g1)
    b = sample(fn, prompt, max_new_tokens=5, config=cfg, generator=g2)
    assert torch.equal(a, b)
