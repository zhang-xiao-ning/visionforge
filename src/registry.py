"""Registry of available experiments.

Each entry maps an experiment name to:
- a factory that returns an nn.Module
- a default learning rate

Adding a new experiment = adding one line here.
"""

from models.deep_convnet import DeepConvNet
from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.vit import ViT
from models.captioning import CaptioningModel

EXPERIMENTS = {
    "mlp": (MLP, 1e-2),
    "shallow_convnet": (ShallowConvNet, 1e-2),
    "deep_convnet": (DeepConvNet, 0.1),
    "vit": (ViT, 3e-4),
    "captioning": (CaptioningModel, 1e-3),
}
