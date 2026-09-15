import os

os.environ["USE_GPU"] = "false"

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def dummy_batch():
    """模拟一个 CIFAR-10 batch：(N, 3, 32, 32) 和 (N,)"""
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    return x, y


class _DummyClassifier(nn.Module):
    """最简单的分类器：flatten + linear。用于测试训练/评估流程。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


@pytest.fixture
def dummy_model():
    """极简分类器，测试训练循环用。"""
    return _DummyClassifier()


@pytest.fixture
def dummy_loader():
    """16 个样本，batch=4，共 4 个 batch。"""
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=4)
