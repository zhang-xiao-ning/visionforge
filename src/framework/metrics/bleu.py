import math
from collections import Counter


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
