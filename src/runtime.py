"""Pure constants + runtime device detection.

`get_device()` resolves on demand; the module-level `device` below is
an import-time snapshot.
"""

import os

import torch

# ---- 纯常量（无副作用）----
PRINT_EVERY = 100
NUM_TRAIN = 49000
BATCH_SIZE = 64
DTYPE = torch.float32
DEFAULT_EXPERIMENT = "vit"


def get_device() -> torch.device:
    """Detect device on demand. Not cached, not at import time."""
    if os.environ.get("USE_GPU", "true").lower() != "true":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# Whether the resolved device is CUDA. Used to decide DataLoader
# num_workers / pin_memory.
USE_CUDA = get_device().type == "cuda"
