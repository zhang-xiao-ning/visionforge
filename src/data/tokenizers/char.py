"""Character-level tokenizer.

Adapter: raw character set → Tokenizer protocol.

Suitable for tiny experiments and unit tests. Not suitable for Chinese
text: a Chinese caption of 20 characters becomes 22 tokens (vs ~10 with
a proper BPE trained on Chinese).
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

    def __init__(self, itos: list[str]) -> None:
        self.itos = itos
        self.stoi = {s: i for i, s in enumerate(itos)}

    # ---------- special ids ----------

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

    # ---------- build ----------

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 2) -> "CharTokenizer":
        """Build a vocabulary from a list of captions."""
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text.lower())

        chars = sorted(c for c, n in counter.items() if n >= min_freq)
        itos = SPECIAL_TOKENS + chars
        return cls(itos)

    # ---------- encode / decode ----------

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(c, self.unk_id) for c in text.lower()]
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

    # ---------- persistence ----------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        with open(path) as f:
            data = json.load(f)
        return cls(data["itos"])
