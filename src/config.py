"""Backward-compatibility shim.

Deprecated: import directly from `runtime` and `experiment.config`
in new code.

Kept for backward compatibility with existing imports.
"""

from experiment.config import TrainConfig  # re-export for compat
from runtime import BATCH_SIZE, DTYPE, NUM_TRAIN, PRINT_EVERY, get_device

dtype = DTYPE
print_every = PRINT_EVERY
device = get_device()

__all__ = [
    "BATCH_SIZE",
    "DTYPE",
    "NUM_TRAIN",
    "PRINT_EVERY",
    "device",
    "dtype",
    "print_every",
    "get_device",
    "TrainConfig",
]
