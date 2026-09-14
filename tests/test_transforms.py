import torch

from data.transforms import cifar10_transform, CIFAR10_MEAN, CIFAR10_STD


def test_cifar10_transform_output():
    from PIL import Image
    import numpy as np

    # 造一张 32x32 RGB 图
    img = Image.fromarray(
        np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    )
    t = cifar10_transform()
    out = t(img)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_cifar10_mean_std_length():
    assert len(CIFAR10_MEAN) == 3
    assert len(CIFAR10_STD) == 3