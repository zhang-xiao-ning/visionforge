"""Pure constants + runtime device detection.

No import side effects. Device is resolved at call time, not import time.
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


# Import-time snapshot. Resolved once, at import.
device = get_device()
