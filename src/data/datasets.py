"""Dataset entry point: name → DataBundle."""

from collections.abc import Callable
from dataclasses import dataclass

from data import cifar10, flickr8k
from data.bundle import DataBundle, DataContext, EvalBundle


@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata for tests / downstream consumers."""

    input_shape: tuple[int, ...]
    num_classes: int


DATASET_REGISTRY: dict[str, Callable[[DataContext], tuple[DataBundle, EvalBundle | None]]] = {
    "cifar10": cifar10.build_bundle,
    "flickr8k": flickr8k.build_bundle,
}

# Metadata for tests that need to know a dataset's shape/classes
# without materializing loaders.
DATASET_INFO: dict[str, DatasetInfo] = {
    "cifar10": DatasetInfo(input_shape=(3, 32, 32), num_classes=10),
}


def build_data(name: str, ctx: DataContext) -> tuple[DataBundle, EvalBundle | None]:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](ctx)
