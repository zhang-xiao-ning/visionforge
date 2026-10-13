"""Generate text from a trained nanoGPT checkpoint.

Usage:
    uv run python scripts/sample_nanogpt.py \
        --checkpoint checkpoints/nanogpt_xxx.pt \
        [--prompt "ROMEO: "] \
        [--max-new-tokens 200] \
        [--strategy greedy|top_k|top_p] \
        [--temperature 0.8] [--top-k 10] [--top-p 0.95]
"""

import argparse
from pathlib import Path

import torch

from application.datasets.tokenizers import CharTokenizer
from application.models import NanoGPT
from framework.inference import SamplingConfig
from framework.runtime import get_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample text from a nanoGPT checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--strategy", type=str, default="top_k", choices=["greedy", "top_k", "top_p"])
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--top-p", type=float, default=0.95)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = NanoGPT(**ckpt["model_init"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    tokenizer = CharTokenizer(
        ckpt["extras"]["itos"],
        lower=ckpt["extras"].get("lower", True),
    )

    if args.strategy == "greedy":
        cfg = SamplingConfig(strategy="greedy")
    elif args.strategy == "top_k":
        cfg = SamplingConfig(
            strategy="top_k", top_k=args.top_k, temperature=args.temperature
        )
    else:
        cfg = SamplingConfig(
            strategy="top_p", top_p=args.top_p, temperature=args.temperature
        )

    ids = tokenizer.encode(args.prompt, add_special=False)
    if not ids:
        raise ValueError("--prompt produced no tokens")
    prompt_ids = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=None,
        config=cfg,
    )
    text = tokenizer.decode(out[0].tolist(), skip_special=False)
    print(text)


if __name__ == "__main__":
    main()
