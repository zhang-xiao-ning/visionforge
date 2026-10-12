"""Adapter: tiktoken.Encoding → Tokenizer protocol.

tiktoken has quirks that the adapter layer smooths over:
- No pad token. We use EOS as pad_id (conventional fallback).
- No unk token. We return unk_id = -1.
- encode() does not add special tokens. We wrap with EOS on both sides
  (GPT-2 has no distinct BOS).

If you need a true pad token, use SentencePiece or Qwen tokenizer
(they have real pad ids).
"""

from pathlib import Path

import tiktoken


class TiktokenTokenizer:
    """Wraps tiktoken.Encoding (implements Tokenizer)."""

    def __init__(self, encoding_name: str = "gpt2") -> None:
        self._enc = tiktoken.get_encoding(encoding_name)
        self._encoding_name = encoding_name
        self._eot = self._enc.eot_token

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def pad_id(self) -> int:
        # tiktoken has no pad; use eos as a conventional fallback.
        return self._eot

    @property
    def bos_id(self) -> int:
        # GPT-2 has no distinct BOS; use eot (same id as eos).
        return self._eot

    @property
    def eos_id(self) -> int:
        return self._eot

    @property
    def unk_id(self) -> int:
        # tiktoken has no unk; caller must handle -1.
        return -1

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = self._enc.encode(text)
        if add_special:
            return [self._eot, *ids, self._eot]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if skip_special:
            ids = [i for i in ids if i != self._eot]
        return self._enc.decode(ids)

    def save(self, path: Path) -> None:
        """tiktoken has no on-disk state; write a small meta file."""
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"type": "tiktoken", "encoding": self._encoding_name},
                indent=2,
            )
        )
