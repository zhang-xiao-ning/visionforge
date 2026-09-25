"""Tokenizer interface.

Every tokenizer implementation must satisfy this Protocol. Upper layers
(models, datasets, tasks) depend only on this protocol, never on concrete
implementations.

Conventions:
- Special-token ids return -1 when the underlying tokenizer has no such
  token (e.g. tiktoken has no unk).
- `encode(text, add_special=True)` wraps with BOS and EOS. If the
  tokenizer lacks BOS or EOS, the corresponding side is skipped.
- `save(path)` format is implementation-defined.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int:
        """Total number of tokens including special tokens."""

    @property
    def pad_id(self) -> int:
        """Padding id, or -1 if the tokenizer has no pad."""

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence id, or -1 if not applicable."""

    @property
    def eos_id(self) -> int:
        """End-of-sequence id, or -1 if not applicable."""

    @property
    def unk_id(self) -> int:
        """Unknown-token id, or -1 if the tokenizer has no unk."""

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Encode text to ids.

        If add_special is True, wrap with BOS and EOS. Tokenizers without
        BOS or EOS skip the corresponding side.
        """

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode ids to text, skipping special tokens by default."""

    def save(self, path: Path) -> None:
        """Persist the tokenizer. Format is implementation-defined."""
