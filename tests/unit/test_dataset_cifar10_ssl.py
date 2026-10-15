"""Tests for the CIFAR-10 self-supervised dataset."""

import numpy as np
import pytest
import torch
from PIL import Image

from application.datasets.cifar10_ssl import (
    CIFAR10SSL,
    VAL_SIZE,
    _ssl_transform,
    _train_val_indices,
)


class _FakeBase:
    """Stand-in for torchvision CIFAR10 with transform=None.

    Returns a random-noise image so stochastic augmentation (crop,
    jitter) actually produces different outputs across calls — a
    constant-colour image would defeat that.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        return Image.fromarray(arr), idx


def test_ssl_transform_output_shape() -> None:
    img = Image.new("RGB", (40, 40), color=(128, 64, 32))
    t = _ssl_transform()
    out = t(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_two_views_are_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same image, same transform object, two calls → different tensors."""

    ds = CIFAR10SSL.__new__(CIFAR10SSL)
    ds.base = _FakeBase(4)
    ds.transform = _ssl_transform()

    v1, v2 = ds[0]
    # Both shape (3, 32, 32) but almost certainly differ due to random aug.
    assert v1.shape == v2.shape == (3, 32, 32)
    assert not torch.equal(v1, v2)


def test_train_val_default_split() -> None:
    train, val = _train_val_indices(50_000, None)
    assert len(train) == 50_000 - VAL_SIZE
    assert len(val) == VAL_SIZE


def test_train_val_num_train_caps_only_train() -> None:
    train, val = _train_val_indices(50_000, 500)
    assert len(train) == 500
    assert len(val) == VAL_SIZE


def test_train_val_no_overlap() -> None:
    for n in (None, 0, 100, 49_000):
        train, val = _train_val_indices(50_000, n)
        assert train.stop <= val.start
