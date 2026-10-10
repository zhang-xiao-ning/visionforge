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

import math
from collections import Counter
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.tokenizers import build_tokenizer
from framework.interfaces import Matric, eval_mode

if TYPE_CHECKING:
    from data.tokenizers import Tokenizer
    from framework.interfaces import DataBundle, EvalBundle


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: list[str],
    references: list[list[str]],
    max_n: int = 4,
) -> float:
    """Corpus BLEU-N.

    Args:
        predictions: one generated string per sample.
        references:  one list of reference strings per sample.
    """
    clipped = [0] * (max_n + 1)
    total = [0] * (max_n + 1)
    gen_len = 0
    ref_len = 0

    for pred, refs in zip(predictions, references, strict=True):
        pred_tokens = pred.lower().split()
        ref_tokens_list = [r.lower().split() for r in refs]
        gen_len += len(pred_tokens)

        if ref_tokens_list:
            ref_len += min(
                (len(r) for r in ref_tokens_list),
                key=lambda length: abs(length - len(pred_tokens)),
            )

        for n in range(1, max_n + 1):
            pred_ng = _ngrams(pred_tokens, n)
            total[n] += sum(pred_ng.values())

            max_ref: Counter[tuple[str, ...]] = Counter()
            for r_tokens in ref_tokens_list:
                r_ng = _ngrams(r_tokens, n)
                for key, count in r_ng.items():
                    if count > max_ref[key]:
                        max_ref[key] = count

            for key, count in pred_ng.items():
                clipped[n] += min(count, max_ref[key])

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        precisions.append(0.0 if total[n] == 0 else clipped[n] / total[n])

    if min(precisions) == 0.0:
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    score = math.exp(log_avg)

    if gen_len < ref_len and gen_len > 0:
        score *= math.exp(1 - ref_len / gen_len)

    return score


class BLEU4(Matric):
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
