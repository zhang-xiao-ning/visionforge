"""Tests for CharTokenizer."""

from pathlib import Path

from data.tokenizer import CharTokenizer


def _make_tokenizer() -> CharTokenizer:
    return CharTokenizer.build(["a cat", "a dog", "hello world"], min_freq=1)


def test_special_ids_are_fixed() -> None:
    tok = _make_tokenizer()
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.unk_id == 3


def test_encode_adds_special_tokens() -> None:
    tok = _make_tokenizer()
    ids = tok.encode("a cat")
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id


def test_encode_without_special() -> None:
    tok = _make_tokenizer()
    ids = tok.encode("abc", add_special=False)
    assert len(ids) == 3


def test_roundtrip() -> None:
    tok = _make_tokenizer()
    text = "a cat"
    assert tok.decode(tok.encode(text)) == text


def test_unknown_char_maps_to_unk() -> None:
    tok = _make_tokenizer()
    ids = tok.encode("z", add_special=False)
    assert ids == [tok.unk_id]


def test_save_and_load(tmp_path: Path) -> None:
    tok = _make_tokenizer()
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    tok2 = CharTokenizer.load(path)
    assert tok.itos == tok2.itos
    assert tok2.vocab_size == tok.vocab_size


def test_decode_skips_special_by_default() -> None:
    tok = _make_tokenizer()
    ids = [tok.bos_id, *tok.encode("a", add_special=False), tok.eos_id]
    assert tok.decode(ids) == "a"


def test_vocab_size_includes_special_tokens() -> None:
    tok = _make_tokenizer()
    # 4 special + unique chars from "a cat"/"a dog"/"hello world"
    # unique chars: space, a, c, d, e, g, h, l, o, r, t, w
    assert tok.vocab_size == 4 + 12
