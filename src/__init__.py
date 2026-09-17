"""visionforge: CIFAR-10 image classification with PyTorch."""

from models.deep_convnet import DeepConvNet
from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.vit import ViT

__version__ = "0.1.0"

__all__ = [
    "MLP",
    "ShallowConvNet",
    "DeepConvNet",
    "ViT",
    "__version__",
]
