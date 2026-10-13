"""Tests for the nanoGPT model."""

import pytest
import torch

from application.models.nanogpt import NanoGPT
from framework.inference import SamplingConfig


def _make_model(vocab: int = 20, block: int = 16) -> NanoGPT:
    return NanoGPT(
        vocab_size=vocab,
        block_size=block,
        d_model=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
    )


def test_forward_shape() -> None:
    model = _make_model()
    ids = torch.randint(0, 20, (2, 8))
    logits = model(ids)
    assert logits.shape == (2, 8, 20)


def test_input_longer_than_block_raises() -> None:
    model = _make_model(block=8)
    ids = torch.randint(0, 20, (1, 16))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(ids)


def test_causal_mask_blocks_future() -> None:
    model = _make_model()
    model.eval()
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[1, 2, 3, 9]])
    with torch.no_grad():
        la = model(a)
        lb = model(b)
    assert torch.allclose(la[:, :3], lb[:, :3], atol=1e-5)


def test_lm_head_tied_to_token_embed() -> None:
    model = _make_model()
    assert model.lm_head.weight is model.token_embed.weight


def test_generate_extends_prompt() -> None:
    model = _make_model()
    prompt = torch.tensor([[1, 2, 3]])
    out = model.generate(prompt, max_new_tokens=5, config=SamplingConfig())
    assert out.shape == (1, 8)
    assert out.dtype == torch.long
    assert out[0, :3].tolist() == [1, 2, 3]


def test_param_groups_default_single_group() -> None:
    model = _make_model()
    groups = model.param_groups()
    assert set(groups.keys()) == {"all"}
