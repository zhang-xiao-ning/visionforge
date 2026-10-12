from collections.abc import Sized
from typing import cast

import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset, Subset

from application.datasets.transforms import cifar10_test_transform, cifar10_train_transform
from framework.interfaces.bundle import DataBundle, DataContext
from framework.runtime import use_cuda
from framework.utils import DATASETS_PATH

# Held-out validation size. Fixed regardless of --num-train, so the
# val curve is comparable across debug runs and full runs.
VAL_SIZE = 1000


def build_datasets() -> tuple[Dataset, Dataset, Dataset]:
    """return (train_set, val_set, test_set)。"""
    train_transform = cifar10_train_transform()
    test_transform = cifar10_test_transform()

    train_set = dset.CIFAR10(
        DATASETS_PATH,
        train=True,
        download=False,
        transform=train_transform,
    )
    val_set = dset.CIFAR10(
        DATASETS_PATH,
        train=True,
        download=False,
        transform=test_transform,
    )
    test_set = dset.CIFAR10(
        DATASETS_PATH,
        train=False,
        download=False,
        transform=test_transform,
    )
    return train_set, val_set, test_set

def _train_val_indices(total: int, num_train: int | None) -> tuple[range, range]:
    """Index ranges over the CIFAR-10 train split.

    The last VAL_SIZE samples are always validation. `num_train` only
    caps how many training samples we use; it never changes val size.
    """
    train_end = total - VAL_SIZE
    n = train_end if num_train is None else min(num_train, train_end)
    return range(n), range(train_end, total)

def build_bundle(ctx: DataContext) -> DataBundle:
    """CIFAR-10 classification bundle."""
    train_set, val_set, test_set = build_datasets()
    total = len(cast(Sized, train_set))

    train_idx, val_idx = _train_val_indices(total, ctx.num_train)
    train_subset = Subset(train_set, train_idx)
    val_subset = Subset(val_set, val_idx)

    strategy = ctx.strategy

    cuda = use_cuda()
    loader_train = DataLoader(
        train_subset,
        batch_size=ctx.batch_size,
        sampler=strategy.make_train_sampler(train_subset),
        num_workers=4 if cuda else 0,
        pin_memory=cuda,
    )
    loader_val = DataLoader(
        val_subset,
        batch_size=ctx.batch_size,
        sampler=strategy.make_val_sampler(val_subset),
        num_workers=4 if cuda else 0,
        pin_memory=cuda,
    )
    loader_test = DataLoader(
        test_set,
        batch_size=ctx.batch_size,
        num_workers=4 if cuda else 0,
        pin_memory=cuda,
    )

    return DataBundle(
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test,
        model_init={"num_classes": 10},
        extras={},
    ), None
