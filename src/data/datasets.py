"""Dataset entry point: name → DataBundle."""

from collections.abc import Callable
from dataclasses import dataclass

from data import cifar10, flickr8k
from framework.interfaces.bundle import DataBundle, DataContext, EvalBundle


@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata for tests / downstream consumers."""

    input_shape: tuple[int, ...]
    num_classes: int


DATASET_REGISTRY: dict[str, Callable[[DataContext], tuple[DataBundle, EvalBundle | None]]] = {
    "cifar10": cifar10.build_bundle,
    "flickr8k": flickr8k.build_bundle,
}

# Metadata for classification datasets. Only datasets whose task is
# classification have a fixed (input_shape, num_classes) — other tasks
# (e.g. captioning) produce variable-length output and don't fit this
# schema. Contract tests iterate this dict, so captioning is correctly
# excluded.
DATASET_INFO: dict[str, DatasetInfo] = {
    "cifar10": DatasetInfo(input_shape=(3, 32, 32), num_classes=10),
}


def build_data(name: str, ctx: DataContext) -> tuple[DataBundle, EvalBundle | None]:
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}")
    return DATASET_REGISTRY[name](ctx)
