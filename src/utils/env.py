import platform
import sys

import torch
import torchvision


def get_env_info():
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


def format_env_info():
    info = get_env_info()
    lines = ["Environment:"]
    for k, v in info.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)
