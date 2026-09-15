import torchvision.datasets as dset

from utils.path import DATASETS_PATH
from data.transforms import cifar10_train_transform, cifar10_test_transform


def build_datasets():
    """返回 (train_set, val_set, test_set)。"""
    train_transform = cifar10_train_transform()
    test_transform = cifar10_test_transform()

    train_set = dset.CIFAR10(
        DATASETS_PATH, train=True, download=False, transform=train_transform,
    )
    val_set = dset.CIFAR10(
        DATASETS_PATH, train=True, download=False, transform=test_transform,
    )
    test_set = dset.CIFAR10(
        DATASETS_PATH, train=False, download=False, transform=test_transform,
    )
    return train_set, val_set, test_set