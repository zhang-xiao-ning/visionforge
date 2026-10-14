"""tiny shakespeare: a character-level language modelling dataset.

Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
Place input.txt at `datasets/tinyshakespeare/input.txt`.

The file is ~1.1 MB, ~1.1M characters. We build a character vocab with
`lower=False` (case matters for the dialogue), split 90/10 train/val,
and expose non-overlapping windows of `block_size` tokens.

`loader_test` reuses the validation split: tiny shakespeare ships with
no official test split, and holding out a third chunk would shrink the
already-small training corpus further. Test perplexity therefore equals
val perplexity — treat it as a sanity check, not a held-out benchmark.

`DataContext.num_train` is ignored: the dataset is small enough to
train on in full and windowed subsampling would distort the language
distribution.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from application.datasets.tokenizers import CharTokenizer
from framework.interfaces.bundle import DataBundle, DataContext
from framework.runtime import use_cuda
from framework.utils import DATASETS_PATH

DEFAULT_PATH = DATASETS_PATH / "tinyshakespeare" / "input.txt"
BLOCK_SIZE = 128
VAL_FRACTION = 0.1

_DOWNLOAD_HINT = (
    "Download input.txt from "
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt "
    "and place it at datasets/tinyshakespeare/input.txt"
)


class ShakespeareDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Non-overlapping windows of `block_size` tokens.

    `__getitem__(i)` returns:
        x = ids[i*B : i*B + B]
        y = ids[i*B + 1 : i*B + 1 + B]
    """

    def __init__(self, ids: torch.Tensor, block_size: int) -> None:
        if len(ids) < block_size + 1:
            raise ValueError(
                f"need at least block_size + 1 = {block_size + 1} tokens, got {len(ids)}"
            )
        self.ids = ids
        self.block_size = block_size

    def __len__(self) -> int:
        return (len(self.ids) - 1) // self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = i * self.block_size
        x = self.ids[start : start + self.block_size]
        y = self.ids[start + 1 : start + 1 + self.block_size]
        return x, y


def build_bundle(ctx: DataContext) -> DataBundle:
    text_path = DEFAULT_PATH
    if not text_path.exists():
        raise FileNotFoundError(f"tiny shakespeare not found at {text_path}. {_DOWNLOAD_HINT}")

    text = text_path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.build([text], min_freq=1, lower=False)
    ids = torch.tensor(tokenizer.encode(text, add_special=False), dtype=torch.long)

    n_val = int(len(ids) * VAL_FRACTION)
    train_ids = ids[:-n_val]
    val_ids = ids[-n_val:]

    train_set = ShakespeareDataset(train_ids, BLOCK_SIZE)
    val_set = ShakespeareDataset(val_ids, BLOCK_SIZE)

    cuda = use_cuda()
    loader_train = DataLoader(
        train_set,
        batch_size=ctx.batch_size,
        sampler=ctx.strategy.make_train_sampler(train_set),
        num_workers=0,
        pin_memory=cuda,
    )
    loader_val = DataLoader(
        val_set,
        batch_size=ctx.batch_size,
        sampler=ctx.strategy.make_val_sampler(val_set),
        num_workers=0,
        pin_memory=cuda,
    )
    loader_test = DataLoader(
        val_set,  # no separate test split for a LM
        batch_size=ctx.batch_size,
        num_workers=0,
        pin_memory=cuda,
    )

    return DataBundle(
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test,
        model_init={"vocab_size": tokenizer.vocab_size, "block_size": BLOCK_SIZE},
        extras={
            "tokenizer_name": "char",
            "itos": tokenizer.itos,
            "lower": tokenizer.lower,
        },
    ), None
