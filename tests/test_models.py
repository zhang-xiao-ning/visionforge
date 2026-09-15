import torch.nn as nn

from models.deep_convnet import DeepConvNet
from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.vit import ViT


def test_mlp_output_shape(dummy_batch):
    x, _ = dummy_batch
    model = MLP()
    out = model(x)
    assert out.shape == (4, 10)


def test_shallow_convnet_output_shape(dummy_batch):
    x, _ = dummy_batch
    model = ShallowConvNet()
    out = model(x)
    assert out.shape == (4, 10)


def test_deep_convnet_output_shape(dummy_batch):
    x, _ = dummy_batch
    model = DeepConvNet()
    out = model(x)
    assert out.shape == (4, 10)


def test_all_models_are_nn_module():
    for cls in (MLP, ShallowConvNet, DeepConvNet):
        model = cls()
        assert isinstance(model, nn.Module)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0


def test_vit_output_shape(dummy_batch):
    x, _ = dummy_batch
    model = ViT()
    out = model(x)
    assert out.shape == (4, 10)
