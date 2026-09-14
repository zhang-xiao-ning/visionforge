# 路径 device hyper-parameter
from dataclasses import dataclass

import torch

USE_GPU = False
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
elif USE_GPU and torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device('cpu')

print('using device:', device)

# Constant to control how frequently we print train loss.
print_every = 100
NUM_TRAIN = 49000
BATCH_SIZE = 64

@dataclass
class TrainConfig:
    """一次训练实验的配置。"""
    experiment: str = "mlp"
    epochs: int = 1
    learning_rate: float = 1e-2
    momentum: float = 0.9
    nesterov: bool = True
    seed: int = 42

