"""Tests for tokenizers."""

from pathlib import Path

import pytest

from application.datasets.tokenizers import (
    CharTokenizer,
    TiktokenTokenizer,
    Tokenizer,
    build_tokenizer,
)

# ---------- CharTokenizer ----------


def _make_char() -> CharTokenizer:
    return CharTokenizer.build(["a cat", "a dog", "hello world"], min_freq=1)


def test_char_special_ids() -> None:
    tok = _make_char()
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3


def test_char_encode_adds_special() -> None:
    tok = _make_char()
    ids = tok.encode("a cat")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_char_roundtrip() -> None:
    tok = _make_char()
    assert tok.decode(tok.encode("a cat")) == "a cat"


def test_char_save_load(tmp_path: Path) -> None:
    tok = _make_char()
    path = tmp_path / "char.json"
    tok.save(path)
    tok2 = CharTokenizer.load(path)
    assert tok.itos == tok2.itos


# ---------- TiktokenTokenizer ----------


def test_tiktoken_vocab_size() -> None:
    tok = TiktokenTokenizer()
    assert tok.vocab_size == 50257


def test_tiktoken_pad_is_eos() -> None:
    tok = TiktokenTokenizer()
    assert tok.pad_id == tok.eos_id


def test_tiktoken_unk_is_minus_one() -> None:
    tok = TiktokenTokenizer()
    assert tok.unk_id == -1


def test_tiktoken_encode_roundtrip() -> None:
    tok = TiktokenTokenizer()
    text = "a cat sitting on a mat"
    ids = tok.encode(text, add_special=False)
    assert tok.decode(ids) == text


def test_tiktoken_encode_with_special() -> None:
    tok = TiktokenTokenizer()
    ids = tok.encode("hello")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_tiktoken_save(tmp_path: Path) -> None:
    tok = TiktokenTokenizer()
    path = tmp_path / "tk.json"
    tok.save(path)
    assert path.exists()


# ---------- Protocol / factory ----------


def test_both_implement_protocol() -> None:
    assert isinstance(_make_char(), Tokenizer)
    assert isinstance(TiktokenTokenizer(), Tokenizer)


def test_factory_returns_tiktoken_by_default() -> None:
    tok = build_tokenizer()
    assert isinstance(tok, TiktokenTokenizer)


def test_factory_returns_char_by_name() -> None:
    tok = build_tokenizer("char", itos=["<pad>", "<bos>", "<eos>", "<unk>", "a"])
    assert isinstance(tok, CharTokenizer)


def test_factory_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tokenizer"):
        build_tokenizer("nope")


def test_char_lower_false_preserves_case() -> None:
    tok = CharTokenizer.build(["Hello World"], min_freq=1, lower=False)
    upper = tok.encode("Hello", add_special=False)
    lower = tok.encode("hello", add_special=False)
    assert upper != lower


def test_char_lower_default_still_lowercases() -> None:
    tok = CharTokenizer.build(["Hello World"], min_freq=1)
    assert tok.lower is True
    # "Hello" and "hello" should encode identically when lower=True
    assert tok.encode("Hello", add_special=False) == tok.encode("hello", add_special=False)


def test_char_save_load_preserves_lower(tmp_path: Path) -> None:
    tok = CharTokenizer.build(["ABC"], min_freq=1, lower=False)
    path = tmp_path / "char.json"
    tok.save(path)
    tok2 = CharTokenizer.load(path)
    assert tok2.lower is False
    assert tok2.itos == tok.itos


def test_char_load_legacy_save_defaults_to_lower(tmp_path: Path) -> None:
    """An old save without a 'lower' key must load with lower=True."""
    path = tmp_path / "legacy.json"
    path.write_text('{"itos": ["<pad>", "<bos>", "<eos>", "<unk>", "a"]}')
    tok = CharTokenizer.load(path)
    assert tok.lower is True
