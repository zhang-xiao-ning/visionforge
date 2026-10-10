"""Corpus BLEU-4.

Minimal implementation: lowercase + whitespace tokenization. Not
guaranteed to match `pycocoevalcap` numbers exactly, but good enough
for tracking a trend during experiments.

A `BLEU4` metric requires:
- a tokenizer (from `DataBundle.extras["tokenizer_name"]`)
- an EvalBundle with an image-level loader (see `Flickr8kImageDataset`)

It calls `model.generate(...)` on each batch, decodes the generated ids,
and computes corpus BLEU-4 against the batch's reference captions.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.tokenizers import build_tokenizer
from framework.interfaces import Metric, eval_mode
from framework.metrics.bleu import corpus_bleu

if TYPE_CHECKING:
    from data.tokenizers import Tokenizer
    from framework.interfaces import DataBundle, EvalBundle


class BLEU4(Metric):
    """Corpus BLEU-4 for image captioning.

    Runs only after training (too slow to run every epoch: generates a
    caption per image). Requires an EvalBundle with image-level batches.
    """

    name = "bleu4"
    higher_is_better = True
    run_every_n_epochs = None

    def __init__(self, tokenizer: "Tokenizer", max_new_tokens: int = 32) -> None:
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @classmethod
    def from_data(
        cls,
        data: "DataBundle",
        eval_data: "EvalBundle | None" = None,
    ) -> "BLEU4":
        tokenizer = build_tokenizer(data.extras["tokenizer_name"])
        return cls(tokenizer=tokenizer)

    def test_loader(
        self,
        data: "DataBundle",
        eval_data: "EvalBundle | None",
    ) -> DataLoader:
        assert eval_data is not None, "BLEU4 requires an EvalBundle"
        return eval_data.loader

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        dtype: torch.dtype,
    ) -> float:
        model.eval()
        predictions: list[str] = []
        references: list[list[str]] = []

        with eval_mode(model):
            for batch in loader:
                images = batch["image"].to(device=device, dtype=dtype)
                generated = model.generate(
                    images,
                    bos_id=self.tokenizer.bos_id,
                    eos_id=self.tokenizer.eos_id,
                    max_new_tokens=self.max_new_tokens,
                )
                for ids in generated:
                    predictions.append(self.tokenizer.decode(ids.tolist()))
                references.extend(batch["references"])

        return corpus_bleu(predictions, references)
