from dataclasses import dataclass

import torch

USE_GPU = True
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device("cuda")
elif USE_GPU and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("using device:", device)

print_every = 100
NUM_TRAIN = 49000
BATCH_SIZE = 64


@dataclass
class TrainConfig:
    experiment: str = "mlp"
    epochs: int = 1
    learning_rate: float = 1e-2
    momentum: float = 0.9
    nesterov: bool = True
    seed: int = 42
    lr_scheduler: str = "none"  # none / step / cosine
    step_size: int = 10  # StepLR 用
    gamma: float = 0.1  # StepLR 用
    early_stop_patience: int = 0  # 0 表示不启用
    amp: bool = False
