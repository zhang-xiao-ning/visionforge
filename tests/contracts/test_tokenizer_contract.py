"""Contract: every tokenizer must satisfy these."""

import pytest

from data.tokenizers import CharTokenizer, Tokenizer, build_tokenizer

_ROUNDTRIP_TEXT = "hello world"


def _make_tokenizer(name: str) -> Tokenizer:
    if name == "char":
        # Build a char vocab that covers the roundtrip text.
        return CharTokenizer.build([_ROUNDTRIP_TEXT], min_freq=1)
    return build_tokenizer(name)


@pytest.fixture(params=["char", "tiktoken"])
def tokenizer(request: pytest.FixtureRequest) -> Tokenizer:
    return _make_tokenizer(request.param)  # type: ignore[no-any-return]


def test_satisfies_protocol(tokenizer: Tokenizer) -> None:
    assert isinstance(tokenizer, Tokenizer)


def test_vocab_size_positive(tokenizer: Tokenizer) -> None:
    assert tokenizer.vocab_size > 0


def test_special_ids_are_valid(tokenizer: Tokenizer) -> None:
    assert tokenizer.pad_id >= -1
    assert tokenizer.bos_id >= -1
    assert tokenizer.eos_id >= -1
    assert tokenizer.unk_id >= -1


def test_encode_returns_ints(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_roundtrip_without_special(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode(_ROUNDTRIP_TEXT, add_special=False)
    decoded = tokenizer.decode(ids)
    assert decoded.lower() == _ROUNDTRIP_TEXT.lower()
