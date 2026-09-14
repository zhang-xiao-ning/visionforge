import torchvision.transforms as T
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from utils.path import DATASETS_PATH
from config import NUM_TRAIN, BATCH_SIZE

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


def build_train_loader():
    train_set = dset.CIFAR10(DATASETS_PATH, train=True, download=False, transform=transform)
    loader = DataLoader(
        train_set, batch_size=BATCH_SIZE,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
    )
    return loader


def build_val_loader():
    val_set = dset.CIFAR10(DATASETS_PATH, train=True, download=False, transform=transform)
    loader = DataLoader(
        val_set, batch_size=BATCH_SIZE,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)),
    )
    return loader


def build_test_loader():
    test_set = dset.CIFAR10(DATASETS_PATH, train=False, download=False, transform=transform)
    loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    return loader