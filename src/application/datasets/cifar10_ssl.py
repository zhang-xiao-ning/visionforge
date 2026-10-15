"""CIFAR-10 for self-supervised learning (DINO-style).

Same train split as cifar10.py, but:
- Labels are discarded.
- Each item returns two independently-augmented views of the same image.

The pair structure is returned as a plain ``(view1, view2)`` tuple;
the default collate stacks each element across the batch, so a batch
is ``(view1_batch, view2_batch)`` with shape ``(B, C, H, W)`` each.
The task unpacks accordingly.

Augmentation is the strong DINO recipe adapted to 32×32 inputs:
RandomResizedCrop + flip + color jitter + occasional grayscale.
Gaussian blur is omitted — at 32×32 it either does nothing or
destroys the signal.
"""

from collections.abc import Sized
from typing import cast

import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset

from application.datasets.transforms import CIFAR10_MEAN, CIFAR10_STD
from framework.interfaces.bundle import DataBundle, DataContext
from framework.runtime import use_cuda
from framework.utils import DATASETS_PATH

VAL_SIZE = 1000  # same held-out size as cifar10.py


def _ssl_transform() -> T.Compose:
    return T.Compose(
        [
            T.RandomResizedCrop(32, scale=(0.4, 1.0), antialias=True),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.2, 0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


class CIFAR10SSL(Dataset[tuple[object, object]]):
    """CIFAR-10 without labels; returns two independently-augmented views."""

    def __init__(self, root: str, train: bool, transform: T.Compose) -> None:
        self.base = dset.CIFAR10(root, train=train, download=False, transform=None)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[object, object]:
        img, _ = self.base[idx]  # PIL.Image; label discarded
        return self.transform(img), self.transform(img)


def _train_val_indices(total: int, num_train: int | None) -> tuple[range, range]:
    """Index ranges over the CIFAR-10 train split.

    Mirrors cifar10._train_val_indices: last VAL_SIZE samples are val,
    num_train caps the training pool, val size is unaffected.
    """
    train_end = total - VAL_SIZE
    n = train_end if num_train is None else min(num_train, train_end)
    return range(n), range(train_end, total)


def build_bundle(ctx: DataContext) -> DataBundle:
    """CIFAR-10 self-supervised bundle."""
    transform = _ssl_transform()
    train_set = CIFAR10SSL(str(DATASETS_PATH), train=True, transform=transform)
    total = len(cast(Sized, train_set))

    train_idx, val_idx = _train_val_indices(total, ctx.num_train)
    train_subset = Subset(train_set, train_idx)
    val_subset = Subset(train_set, val_idx)

    cuda = use_cuda()
    loader_train = DataLoader(
        train_subset,
        batch_size=ctx.batch_size,
        sampler=ctx.strategy.make_train_sampler(train_subset),
        num_workers=4 if cuda else 0,
        pin_memory=cuda,
    )
    loader_val = DataLoader(
        val_subset,
        batch_size=ctx.batch_size,
        sampler=ctx.strategy.make_val_sampler(val_subset),
        num_workers=4 if cuda else 0,
        pin_memory=cuda,
    )
    loader_test = DataLoader(
        val_subset,  # no separate test split for SSL
        batch_size=ctx.batch_size,
        num_workers=4 if cuda else 0,
        pin_memory=cuda,
    )

    return DataBundle(
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test,
        model_init={
            "img_size": 32,
            "patch_size": 4,
            "in_channels": 3,
            "embed_dim": 192,
            "depth": 4,
            "num_heads": 6,
            "head_out": 1024,
        },
        extras={},
    ), None
