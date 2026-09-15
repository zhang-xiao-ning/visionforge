import pytest
import torch


@pytest.fixture
def dummy_batch():
    """模拟一个 CIFAR-10 batch：(N, 3, 32, 32) 和 (N,)"""
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    return x, y
