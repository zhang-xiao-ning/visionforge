import numpy as np
import torch
from PIL import Image

from data.transforms import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    cifar10_test_transform,
    cifar10_train_transform,
)


def _make_image():
    return Image.fromarray(
        np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    )


def test_train_transform_output():
    img = _make_image()
    t = cifar10_train_transform()
    out = t(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_test_transform_output():
    img = _make_image()
    t = cifar10_test_transform()
    out = t(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_cifar10_mean_std_length():
    assert len(CIFAR10_MEAN) == 3
    assert len(CIFAR10_STD) == 3