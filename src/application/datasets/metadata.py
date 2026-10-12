"""Static metadata for datasets (used by contract tests)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    """Static metadata for tests / downstream consumers."""

    input_shape: tuple[int, ...]
    num_classes: int


# Metadata for classification datasets. Only datasets whose task is
# classification have a fixed (input_shape, num_classes) — other tasks
# (e.g. captioning) produce variable-length output and don't fit this
# schema.
DATASET_INFO: dict[str, DatasetInfo] = {
    "cifar10": DatasetInfo(input_shape=(3, 32, 32), num_classes=10),
}
