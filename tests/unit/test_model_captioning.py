"""Tests for the captioning model."""

import pytest
import torch

from models.captioning import CaptioningModel


def _make_model(vocab_size: int = 50, pad_id: int = 0, max_len: int = 16) -> CaptioningModel:
    return CaptioningModel(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_len=max_len,
        d_model=64,
        num_heads=4,
        encoder_layers=2,
        decoder_layers=2,
        dim_feedforward=128,
        image_size=64,
        patch_size=16,
    )


def test_forward_output_shape() -> None:
    model = _make_model()
    images = torch.randn(2, 3, 64, 64)
    input_ids = torch.randint(0, 50, (2, 8))
    logits = model(images, input_ids)
    assert logits.shape == (2, 8, 50)


def test_forward_single_sample() -> None:
    model = _make_model()
    images = torch.randn(1, 3, 64, 64)
    input_ids = torch.randint(0, 50, (1, 4))
    logits = model(images, input_ids)
    assert logits.shape == (1, 4, 50)


def test_input_length_exceeds_max_len_raises() -> None:
    model = _make_model(max_len=8)
    images = torch.randn(1, 3, 64, 64)
    input_ids = torch.randint(0, 50, (1, 16))
    with pytest.raises(ValueError, match="exceeds max_len"):
        model(images, input_ids)


def test_causal_mask_blocks_future() -> None:
    """Changing the last input token should not affect earlier logits."""
    torch.manual_seed(0)
    model = _make_model()
    model.eval()

    images = torch.randn(1, 3, 64, 64)
    ids_a = torch.tensor([[1, 2, 3, 4, 5]])
    ids_b = torch.tensor([[1, 2, 3, 4, 9]])  # last token different

    with torch.no_grad():
        logits_a = model(images, ids_a)
        logits_b = model(images, ids_b)

    # positions 0..3 should be identical
    assert torch.allclose(logits_a[:, :4], logits_b[:, :4], atol=1e-5)


def test_generate_returns_bos_at_least() -> None:
    model = _make_model()
    images = torch.randn(1, 3, 64, 64)
    out = model.generate(images, bos_id=1, eos_id=2, max_new_tokens=5)
    assert out.shape[0] == 1
    assert out[0, 0].item() == 1  # starts with BOS


def test_generate_respects_max_new_tokens() -> None:
    torch.manual_seed(0)
    model = _make_model(max_len=32)
    model.eval()
    images = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        out = model.generate(images, bos_id=1, eos_id=2, max_new_tokens=5)

    # BOS + at most 5 new tokens
    assert out.shape[1] <= 6
    assert out.shape[1] >= 2
    assert out[0, 0].item() == 1  # starts with BOS


def test_lm_head_tied_to_token_embed() -> None:
    model = _make_model()
    # 默认 tie_weights=True
    assert model.lm_head.weight is model.token_embed.weight


def test_tie_weights_false_untied() -> None:
    model = CaptioningModel(
        vocab_size=50,
        pad_id=0,
        d_model=64,
        num_heads=4,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=128,
        image_size=64,
        patch_size=16,
        tie_weights=False,
    )
    assert model.lm_head.weight is not model.token_embed.weight


def test_generate_forces_eos_after_finish() -> None:
    """After a sample emits EOS, remaining positions are EOS, not garbage.

    Regression test: without the mask, samples in a batch that finish
    early keep generating arbitrary ids after EOS, polluting BLEU.
    """
    model = _make_model(vocab_size=10, pad_id=0, max_len=16)
    eos_id = 2
    step = {"n": 0}

    def fake_forward(images, input_ids):
        step["n"] += 1
        B, L = input_ids.shape
        logits = torch.zeros(B, L, 10, device=images.device)
        if step["n"] == 1:
            # 样本 0 第一步就 EOS；样本 1 预测 token 5
            logits[0, -1, eos_id] = 1.0
            logits[1, -1, 5] = 1.0
        else:
            # 之后所有样本都预测 token 5（非 EOS）
            logits[:, -1, 5] = 1.0
        return logits

    model.forward = fake_forward  # type: ignore[method-assign]

    images = torch.randn(2, 3, 64, 64)
    out = model.generate(images, bos_id=1, eos_id=eos_id, max_new_tokens=3)

    # 样本 0: [BOS, EOS, EOS, EOS]，EOS 后是 EOS 填充
    # 样本 1: [BOS, 5, 5, 5]，永不结束
    assert out[0].tolist() == [1, eos_id, eos_id, eos_id]
    assert out[1].tolist() == [1, 5, 5, 5]
