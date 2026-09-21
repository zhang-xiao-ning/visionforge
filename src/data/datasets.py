from torch.utils.data import DataLoader, Subset

from config import BATCH_SIZE, NUM_TRAIN, device
from data import cifar10
from training.strategy import SingleDeviceStrategy, TrainingStrategy

DATASET_REGISTRY = {
    "cifar10": cifar10.build_datasets,
}


def build_loaders(
    name: str = "cifar10",
    batch_size: int = BATCH_SIZE,
    num_train: int = NUM_TRAIN,
    strategy: TrainingStrategy | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if strategy is None:
        strategy = SingleDeviceStrategy()

    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")

    train_set, val_set, test_set = DATASET_REGISTRY[name]()
    total = len(train_set)
    use_cuda = device.type == "cuda"

    train_subset = Subset(train_set, range(num_train))
    val_subset = Subset(val_set, range(num_train, total))

    loader_train = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=strategy.make_train_sampler(train_subset),
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    loader_val = DataLoader(
        val_subset,
        batch_size=batch_size,
        sampler=strategy.make_val_sampler(val_subset),
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    loader_test = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    return loader_train, loader_val, loader_test
