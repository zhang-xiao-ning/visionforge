from torch.utils.data import DataLoader, sampler

from config import NUM_TRAIN, BATCH_SIZE, device
from data import cifar10


DATASET_REGISTRY = {
    "cifar10": cifar10.build_datasets,
}


def build_loaders(name="cifar10", batch_size=BATCH_SIZE, num_train=NUM_TRAIN):
    if name not in DATASET_REGISTRY:
        raise ValueError("Unknown dataset: %s" % name)

    train_set, val_set, test_set = DATASET_REGISTRY[name]()
    total = len(train_set)
    use_cuda = device.type == "cuda"

    loader_train = DataLoader(
        train_set, batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(range(num_train)),
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    loader_val = DataLoader(
        val_set, batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(range(num_train, total)),
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    loader_test = DataLoader(
        test_set, batch_size=batch_size,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    return loader_train, loader_val, loader_test