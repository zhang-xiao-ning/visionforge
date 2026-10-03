from collections.abc import Sized
from typing import cast

import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset, Subset

from data.bundle import DataBundle, DataContext
from data.transforms import cifar10_test_transform, cifar10_train_transform
from runtime import NUM_TRAIN, USE_CUDA
from training.strategy import SingleDeviceStrategy
from utils.path import DATASETS_PATH


def build_datasets() -> tuple[Dataset, Dataset, Dataset]:
    """返回 (train_set, val_set, test_set)。"""
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


def build_bundle(ctx: DataContext) -> DataBundle:
    """CIFAR-10 classification bundle."""
    train_set, val_set, test_set = build_datasets()
    total = len(cast(Sized, train_set))
    num_train = ctx.num_train if ctx.num_train is not None else NUM_TRAIN

    strategy = ctx.strategy if ctx.strategy is not None else SingleDeviceStrategy()

    train_subset = Subset(train_set, range(num_train))
    val_subset = Subset(val_set, range(num_train, total))

    loader_train = DataLoader(
        train_subset,
        batch_size=ctx.batch_size,
        sampler=strategy.make_train_sampler(train_subset),
        num_workers=4 if USE_CUDA else 0,
        pin_memory=USE_CUDA,
    )
    loader_val = DataLoader(
        val_subset,
        batch_size=ctx.batch_size,
        sampler=strategy.make_val_sampler(val_subset),
        num_workers=4 if USE_CUDA else 0,
        pin_memory=USE_CUDA,
    )
    loader_test = DataLoader(
        test_set,
        batch_size=ctx.batch_size,
        num_workers=4 if USE_CUDA else 0,
        pin_memory=USE_CUDA,
    )

    return DataBundle(
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test,
        model_init={"num_classes": 10},
        extras={},
    ), None
