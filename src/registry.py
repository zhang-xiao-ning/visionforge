"""Registry of available experiments.

Each entry maps an experiment name to:
- data:  which dataset to build
- model: which model class to instantiate from the data
- task:  which Task class to instantiate from the data
- lr:    default learning rate
- category: for filtering in tests

Adding a new experiment = adding one dict entry here.
"""

from models.captioning import CaptioningModel
from models.vit import ViT
from tasks.captioning import CaptioningTask
from tasks.classification import ClassificationTask

EXPERIMENTS: dict[str, dict] = {
    "vit": {
        "data": "cifar10",
        "model": ViT,
        "task": ClassificationTask,
        "lr": 3e-4,
        "category": "classification",
    },
    "captioning": {
        "data": "flickr8k",
        "model": CaptioningModel,
        "task": CaptioningTask,
        "lr": 1e-3,
        "category": "captioning",
    },
}
