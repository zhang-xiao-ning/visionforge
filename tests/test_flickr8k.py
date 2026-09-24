"""Tests for Flickr8k data pipeline."""

from pathlib import Path

import torch
from PIL import Image

from data.flickr8k import (
    Flickr8kDataset,
    _eval_transform,
    load_captions,
    load_split,
    make_collate_fn,
)
from data.tokenizer import CharTokenizer


def _make_fixture(tmp_path: Path) -> tuple[Path, dict[str, list[str]]]:
    """Create a minimal fake Flickr8k-like structure."""
    image_dir = tmp_path / "Flicker8k_Dataset"
    image_dir.mkdir()

    # 2 fake images
    for name in ["a.jpg", "b.jpg"]:
        Image.new("RGB", (300, 300), color=(128, 128, 128)).save(image_dir / name)

    captions = {"a.jpg": ["a cat", "a small cat"], "b.jpg": ["a dog"]}
    return image_dir, captions


def test_load_captions_parses_file(tmp_path: Path) -> None:
    f = tmp_path / "Flickr8k.token.txt"
    f.write_text("a.jpg#0\ta cat\na.jpg#1\ta small cat\nb.jpg#0\ta dog\n")
    captions = load_captions(f)
    assert captions["a.jpg"] == ["a cat", "a small cat"]
    assert captions["b.jpg"] == ["a dog"]


def test_load_split_reads_lines(tmp_path: Path) -> None:
    f = tmp_path / "train.txt"
    f.write_text("a.jpg\nb.jpg\n")
    assert load_split(f) == ["a.jpg", "b.jpg"]


def test_dataset_len_is_num_pairs(tmp_path: Path) -> None:
    image_dir, captions = _make_fixture(tmp_path)
    tok = CharTokenizer.build([c for cs in captions.values() for c in cs], min_freq=1)
    ds = Flickr8kDataset(image_dir, ["a.jpg", "b.jpg"], captions, tok, _eval_transform())
    # a.jpg has 2 captions, b.jpg has 1
    assert len(ds) == 3


def test_dataset_item_shapes(tmp_path: Path) -> None:
    image_dir, captions = _make_fixture(tmp_path)
    tok = CharTokenizer.build([c for cs in captions.values() for c in cs], min_freq=1)
    ds = Flickr8kDataset(image_dir, ["a.jpg"], captions, tok, _eval_transform())
    item = ds[0]
    assert item["image"].shape == (3, 224, 224)
    assert item["input_ids"].dtype == torch.long
    assert item["target_ids"].dtype == torch.long
    # input_ids and target_ids are the same length
    assert item["input_ids"].shape == item["target_ids"].shape


def test_input_target_are_shifted(tmp_path: Path) -> None:
    image_dir, captions = _make_fixture(tmp_path)
    tok = CharTokenizer.build(["a cat"], min_freq=1)
    ds = Flickr8kDataset(image_dir, ["a.jpg"], {"a.jpg": ["a cat"]}, tok, _eval_transform())
    item = ds[0]

    # input starts with BOS, target ends with EOS
    assert item["input_ids"][0].item() == tok.bos_id
    assert item["target_ids"][-1].item() == tok.eos_id
    # length is one less than full encoding
    full = tok.encode("a cat")
    assert item["input_ids"].shape[0] == len(full) - 1


def test_collate_pads_to_max_length(tmp_path: Path) -> None:
    image_dir, captions = _make_fixture(tmp_path)
    tok = CharTokenizer.build([c for cs in captions.values() for c in cs], min_freq=1)
    ds = Flickr8kDataset(image_dir, ["a.jpg", "b.jpg"], captions, tok, _eval_transform())
    collate = make_collate_fn(tok.pad_id)
    batch = collate([ds[0], ds[1], ds[2]])

    assert batch["image"].shape == (3, 3, 224, 224)
    assert batch["input_ids"].shape == batch["target_ids"].shape
    # padded positions should be pad_id
    assert (batch["input_ids"][:, -1] == tok.pad_id).any()


def test_max_len_truncates(tmp_path: Path) -> None:
    image_dir, _ = _make_fixture(tmp_path)
    tok = CharTokenizer.build(["a"], min_freq=1)
    long_caption = "a" * 200
    ds = Flickr8kDataset(
        image_dir,
        ["a.jpg"],
        {"a.jpg": [long_caption]},
        tok,
        _eval_transform(),
        max_len=16,
    )
    item = ds[0]
    assert item["input_ids"].shape[0] <= 15  # max_len - 1
