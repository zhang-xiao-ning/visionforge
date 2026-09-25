"""Flickr8k dataset for image captioning.

Layout expected:
    datasets/Flicker8k_Dataset/           # 8091 images
    datasets/Flickr8k_text/
        Flickr8k.token.txt                # 40460 caption lines
        Flickr_8k.trainImages.txt
        Flickr_8k.devImages.txt
        Flickr_8k.testImages.txt

Note the misspelling: "Flicker8k" (image dir) vs "Flickr8k" (text dir).
This is the original author's typo, preserved to match the released zip.
"""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data.tokenizers import Tokenizer

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IGNORE_INDEX = -100  # PyTorch default; not a valid token id


def _train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_captions(captions_file: Path) -> dict[str, list[str]]:
    """Parse Flickr8k.token.txt → {image_name: [caption1, caption2, ...]}."""
    captions: dict[str, list[str]] = defaultdict(list)
    with open(captions_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, caption = line.split("\t", 1)
            image_name = key.split("#")[0]
            captions[image_name].append(caption)
    return dict(captions)


def load_split(split_file: Path) -> list[str]:
    """Read a list of image names (e.g. Flickr_8k.trainImages.txt)."""
    with open(split_file) as f:
        return [line.strip() for line in f if line.strip()]


class Flickr8kDataset(Dataset[dict[str, Any]]):
    """One sample = one (image, caption) pair.

    Returns dict with:
    - image:      (3, H, W) float tensor
    - input_ids:  1D long tensor, [BOS, c1, c2, ..., cN]
    - target_ids: 1D long tensor, [c1, c2, ..., cN, EOS]
    """

    def __init__(
        self,
        image_dir: Path,
        image_names: list[str],
        captions: dict[str, list[str]],
        tokenizer: Tokenizer,
        transform: transforms.Compose,
        max_len: int = 64,
    ) -> None:
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

        # flatten: one entry per (image, caption) pair
        self.items: list[tuple[str, str]] = []
        for name in image_names:
            for caption in captions.get(name, []):
                self.items.append((name, caption))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_name, caption = self.items[idx]

        image = Image.open(self.image_dir / image_name).convert("RGB")
        image_tensor = self.transform(image)

        ids = self.tokenizer.encode(caption)
        # truncate to max_len (keep EOS)
        if len(ids) > self.max_len:
            ids = ids[: self.max_len - 1] + [self.tokenizer.eos_id]

        # input_ids: [BOS, c1, ..., cN]   (feed to decoder)
        # target_ids: [c1, ..., cN, EOS]  (predict)
        input_ids = ids[:-1]
        target_ids = ids[1:]

        return {
            "image": image_tensor,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
        }


def make_collate_fn(pad_id: int) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    """Return a collate_fn that pads input_ids / target_ids to the same length."""

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = torch.stack([b["image"] for b in batch])
        max_len = max(len(b["input_ids"]) for b in batch)

        # input_ids: padded with pad_id (decoder uses it for key_padding_mask)
        # target_ids: padded with -100 (loss uses ignore_index=-100)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        target_ids = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)

        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = b["input_ids"]
            target_ids[i, :n] = b["target_ids"]

        return {
            "image": images,
            "input_ids": input_ids,
            "target_ids": target_ids,
        }

    return collate_fn


def build_captioning_loaders(
    root: Path,
    tokenizer: Tokenizer,
    batch_size: int = 32,
    num_workers: int = 0,
    max_len: int = 64,
    image_size: int = 224,
    num_train: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test loaders from a Flickr8k root dir."""
    image_dir = root / "Flicker8k_Dataset"
    text_dir = root / "Flickr8k_text"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not text_dir.exists():
        raise FileNotFoundError(f"Text dir not found: {text_dir}")

    captions = load_captions(text_dir / "Flickr8k.token.txt")
    train_names = load_split(text_dir / "Flickr_8k.trainImages.txt")
    val_names = load_split(text_dir / "Flickr_8k.devImages.txt")
    test_names = load_split(text_dir / "Flickr_8k.testImages.txt")

    if num_train is not None:
        train_names = train_names[:num_train]

    train_tf = _train_transform(image_size)
    eval_tf = _eval_transform(image_size)

    train_set = Flickr8kDataset(image_dir, train_names, captions, tokenizer, train_tf, max_len)
    val_set = Flickr8kDataset(image_dir, val_names, captions, tokenizer, eval_tf, max_len)
    test_set = Flickr8kDataset(image_dir, test_names, captions, tokenizer, eval_tf, max_len)

    collate = make_collate_fn(tokenizer.pad_id)

    loader_train = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    loader_val = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )
    loader_test = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
    )

    return loader_train, loader_val, loader_test
