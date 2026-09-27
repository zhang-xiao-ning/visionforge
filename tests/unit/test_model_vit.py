"""Tests for the ViT model."""

import torch.nn as nn

from models.vit import ViT


def test_output_shape(dummy_batch):
    x, _ = dummy_batch
    model = ViT()
    out = model(x)
    assert out.shape == (4, 10)


def test_is_nn_module():
    model = ViT()
    assert isinstance(model, nn.Module)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
