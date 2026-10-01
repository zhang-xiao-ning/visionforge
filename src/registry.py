"""Registry of available experiments.

Each entry is an `Experiment` object: which model class, which task
class, which dataset key, plus defaults. Stages are NOT here — they
are provided at run time (CLI or YAML).
"""

from experiment.spec import Experiment, TrainConfig
from models.captioning import CaptioningModel
from models.vit import ViT
from tasks.captioning import CaptioningTask
from tasks.classification import ClassificationTask

EXPERIMENTS: dict[str, Experiment] = {
    "vit": Experiment(
        model=ViT,
        task=ClassificationTask,
        data="cifar10",
        config=TrainConfig(learning_rate=3e-4),
        category="classification",
    ),
    "captioning": Experiment(
        model=CaptioningModel,
        task=CaptioningTask,
        data="flickr8k",
        config=TrainConfig(learning_rate=1e-3),
        category="captioning",
    ),
}
