"""Autoregressive sampling, model-agnostic.

Design:

- `SamplingConfig` — frozen configuration for one strategy. Validated
  at construction; invalid combinations raise rather than being
  silently ignored.
- `apply_sampling` — single-step pure function. `(B, V) logits -> (B,)`
  token ids. Exported so callers with their own decode loop can reuse
  just the strategy.
- `sample` — drives the decode loop, threading state through a
  caller-supplied `logits_fn`.

Scope:

- `LogitsFn` is called with the *full* sequence every step. KV cache /
  incremental decoding is out of scope; a future variant can add a
  `past` argument without breaking this contract.
- Randomness is controllable via `torch.Generator`. `None` means the
  global RNG.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F

LogitsFn = Callable[[torch.Tensor], torch.Tensor]
"""Maps (B, L) long ids to (B, L, V) float logits."""

SamplingStrategy = Literal["greedy", "top_k", "top_p"]


@dataclass(frozen=True)
class SamplingConfig:
    """One sampling strategy, fully specified.

    Validation is strict on purpose: `greedy` with a stray `top_k=50`
    is a bug, not a hint to ignore.
    """

    strategy: SamplingStrategy = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

        if self.strategy == "greedy":
            if self.top_k is not None or self.top_p is not None:
                raise ValueError("greedy sampling does not accept top_k / top_p")
            if self.temperature != 1.0:
                raise ValueError("greedy sampling does not use temperature; leave it at 1.0")
        elif self.strategy == "top_k":
            if self.top_k is None or self.top_k <= 0:
                raise ValueError(f"top_k sampling requires top_k > 0, got {self.top_k}")
            if self.top_p is not None:
                raise ValueError("top_k sampling does not accept top_p")
        elif self.strategy == "top_p":
            if self.top_p is None or not (0.0 < self.top_p <= 1.0):
                raise ValueError(f"top_p sampling requires 0 < top_p <= 1, got {self.top_p}")
            if self.top_k is not None:
                raise ValueError("top_p sampling does not accept top_k")
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


def apply_sampling(
    logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """Pick one token per row. `logits`: (B, V) float -> (B,) long."""
    if config.strategy == "greedy":
        return cast(torch.LongTensor, logits.argmax(dim=-1))

    scaled = logits / config.temperature

    if config.strategy == "top_k":
        assert config.top_k is not None  # validated in __post_init__
        k = min(config.top_k, scaled.size(-1))
        topk_vals, _ = scaled.topk(k, dim=-1)
        threshold = topk_vals[..., -1, None]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

    elif config.strategy == "top_p":
        assert config.top_p is not None  # validated in __post_init__
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = probs.cumsum(dim=-1)
        mask = cumprobs > config.top_p
        mask[..., 0] = False  # always keep at least the top token
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    else:  # pragma: no cover — config validation prevents this
        raise ValueError(f"Unknown strategy: {config.strategy}")

    probs = F.softmax(scaled, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return cast(torch.LongTensor, next_token.squeeze(-1))


@torch.no_grad()
def sample(
    logits_fn: LogitsFn,
    prompt: torch.LongTensor,
    *,
    max_new_tokens: int,
    eos_id: int | None = None,
    config: SamplingConfig = SamplingConfig(),
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """Autoregressively extend `prompt` by at most `max_new_tokens`.

    Contract:
        - `max_new_tokens < 1` returns `prompt` unchanged and never
          calls `logits_fn`.
        - `logits_fn` is called exactly once per generated token, so
          the call count equals the number of tokens appended.
        - `eos_id is None`: no early stop, always `max_new_tokens` tokens.
        - `eos_id` given: a row that emits `eos_id` keeps that EOS in
          the output and is filled with `eos_id` on subsequent steps.
          The loop breaks early once every row has finished.

    Returns:
        (B, L0 + K) long tensor, `K <= max_new_tokens`.
    """
    if max_new_tokens < 1:
        return prompt

    B = prompt.size(0)
    generated: torch.Tensor = prompt

    finished: torch.Tensor | None = None
    if eos_id is not None:
        finished = torch.zeros(B, dtype=torch.bool, device=prompt.device)

    for _ in range(max_new_tokens):
        logits = logits_fn(generated)  # (B, L, V)
        next_token = apply_sampling(logits[:, -1, :], config, generator=generator)

        if eos_id is not None:
            assert finished is not None
            next_token = cast(
                torch.LongTensor,
                next_token.masked_fill(finished, eos_id),
            )

        generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=1)

        if eos_id is not None:
            assert finished is not None
            finished = finished | (next_token == eos_id)
            if finished.all():
                break

    return cast(torch.LongTensor, generated)
