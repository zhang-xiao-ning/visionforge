"""Registry of available experiments.

Each entry maps an experiment name to:
- a factory that returns a nn.Module
- a default learning rate

Adding a new experiment = adding one line here.
"""

from models.captioning import CaptioningModel
from models.vit import ViT

EXPERIMENTS = {
    "vit": (ViT, 3e-4),
    "captioning": (CaptioningModel, 1e-3),
}
