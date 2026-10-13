"""Character-level tokenizer.

Adapter: raw character set → Tokenizer protocol.

Suitable for tiny experiments and unit tests. Not suitable for Chinese
text: a Chinese caption of 20 characters becomes 22 tokens (vs ~10 with
a proper BPE trained on Chinese).

Case handling is controlled by `lower` (default True, matching the
original behaviour). Set `lower=False` for corpora where case carries
information, e.g. tiny shakespeare.
"""

import json
from collections import Counter
from pathlib import Path

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class CharTokenizer:
    """Maps characters to integer ids and back (implements Tokenizer)."""

    def __init__(self, itos: list[str], lower: bool = True) -> None:
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(itos)}
        self.lower = lower

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @classmethod
    def build(
        cls,
        texts: list[str],
        min_freq: int = 2,
        lower: bool = True,
    ) -> "CharTokenizer":
        """Build a vocabulary from a list of strings."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text.lower() if lower else text)

        chars = sorted(c for c, n in counter.items() if n >= min_freq)
        itos = SPECIAL_TOKENS + chars
        return cls(itos, lower=lower)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        src = text.lower() if self.lower else text
        ids = [self.stoi.get(c, self.unk_id) for c in src]
        if add_special:
            return [self.bos_id, *ids, self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        out: list[str] = []
        for i in ids:
            if skip_special and i in specials:
                continue
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return "".join(out)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {"itos": self.itos, "lower": self.lower}, f, ensure_ascii=False, indent=2
            )

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        with open(path) as f:
            data = json.load(f)
        # Backward compat: old saves had no "lower" key.
        return cls(data["itos"], lower=data.get("lower", True))
