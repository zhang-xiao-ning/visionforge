"""Tokenizer package: protocol + adapters + factory.

Usage:
    from data.tokenizers import build_tokenizer
    tok = build_tokenizer("tiktoken")

To add a new backend:
    1. Create a new module in this package (e.g. `sentencepiece_bpe.py`).
    2. Implement the Tokenizer protocol (all attributes and methods).
    3. Register it in `_REGISTRY` below.
"""

from typing import Any

from data.tokenizers.base import Tokenizer
from data.tokenizers.char import CharTokenizer
from data.tokenizers.tiktoken_bpe import TiktokenTokenizer

_REGISTRY: dict[str, type] = {
    "char": CharTokenizer,
    "tiktoken": TiktokenTokenizer,
}

__all__ = ["Tokenizer", "build_tokenizer", "CharTokenizer", "TiktokenTokenizer"]


def build_tokenizer(name: str = "tiktoken", **kwargs: Any) -> Tokenizer:
    """Factory. Builds a tokenizer by name.

    Args:
        name: backend name (see _REGISTRY).
        **kwargs: forwarded to the implementation's __init__.
    """
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown tokenizer '{name}'. Known: {known}")
    return _REGISTRY[name](**kwargs)
