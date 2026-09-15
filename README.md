# My Test Project

CIFAR-10 image classification with PyTorch. A small, clean, extensible
training framework that supports multiple models (MLP / ConvNet / ViT),
resumable training, and reproducible experiments.

## Requirements

- Python 3.12
- PyTorch 2.2
- (optional) CUDA 12.1 for GPU

## Using as a Library

```python
from my_test_project import MLP, ViT
model = ViT()

## Installation

```bash
git clone <repo-url>
cd my_test_project
pip install -e ".[dev]"

