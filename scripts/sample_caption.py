"""Generate a caption for an image using a trained checkpoint.

Usage:
    uv run python scripts/sample_caption.py \
        --checkpoint checkpoints/captioning_xxx.pt \
        --image path/to/image.jpg \
        [--max-new-tokens 32]
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from data.flickr8k import _eval_transform
from data.tokenizers import build_tokenizer
from models.captioning import CaptioningModel
from runtime import device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a caption for an image.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Reconstruct tokenizer from checkpoint metadata
    tokenizer_name = ckpt["extras"]["tokenizer_name"]
    tokenizer = build_tokenizer(tokenizer_name)

    # Reconstruct model from checkpoint metadata
    model = CaptioningModel(**ckpt["model_init"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    transform = _eval_transform()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        generated = model.generate(
            image_tensor,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=args.max_new_tokens,
        )

    caption = tokenizer.decode(generated[0].tolist())
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Image:      {image_path}")
    print(f"Caption:    {caption}")


if __name__ == "__main__":
    main()