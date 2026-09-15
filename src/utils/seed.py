import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    # cudnn 会引入不确定性，但对小网络影响不大
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
