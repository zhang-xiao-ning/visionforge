"""VisionForge: image classification and captioning with PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("visionforge")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
