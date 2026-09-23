"""VisionForge: CIFAR-10 image classification with PyTorch."""

from importlib.metadata import PackageNotFoundError, version

from models.deep_convnet import DeepConvNet
from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.vit import ViT

try:
    __version__ = version("visionforge")
except PackageNotFoundError:
    # 源码运行时（未安装），回退到占位版本
    __version__ = "0.0.0+unknown"

__all__ = [
    "MLP",
    "ShallowConvNet",
    "DeepConvNet",
    "ViT",
    "__version__",
]
