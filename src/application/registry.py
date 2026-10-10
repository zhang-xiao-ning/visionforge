"""Registry of available experiments."""

from application.metrics import BLEU4
from application.models import MLP, CaptioningModel, MambaClassifier, ViT
from application.tasks.captioning import CaptioningTask
from application.tasks.classification import ClassificationTask
from framework.experiment.spec import Experiment, TrainConfig
from framework.metrics import Accuracy, CrossEntropy, Perplexity
from framework.registry import EXPERIMENTS

# Each entry is written out in full, even though the classification
# entries share most fields. This is deliberate: the registry is a
# configuration file, and a small amount of duplication keeps each
# experiment self-contained and independently editable. Extracting a
# helper would force every future divergence (a different metric, a
# different dataset) to flow through the helper's signature.


EXPERIMENTS.register(
    "mlp",
    Experiment(
        model=MLP,
        task=ClassificationTask,
        data="cifar10",
        metrics=[Accuracy, CrossEntropy],
        primary_metric="acc",
        config=TrainConfig(learning_rate=1e-2),
        category="classification",
    ),
)
EXPERIMENTS.register(
    "vit",
    Experiment(
        model=ViT,
        task=ClassificationTask,
        data="cifar10",
        metrics=[Accuracy, CrossEntropy],
        primary_metric="acc",
        config=TrainConfig(learning_rate=3e-4),
        category="classification",
    ),
)
EXPERIMENTS.register(
    "mamba",
    Experiment(
        model=MambaClassifier,
        task=ClassificationTask,
        data="cifar10",
        metrics=[Accuracy, CrossEntropy],
        primary_metric="acc",
        config=TrainConfig(learning_rate=3e-4, batch_size=64),
        category="classification",
    ),
)
EXPERIMENTS.register(
    "captioning",
    Experiment(
        model=CaptioningModel,
        task=CaptioningTask,
        data="flickr8k",
        metrics=[Perplexity, BLEU4],
        primary_metric="perplexity",
        config=TrainConfig(learning_rate=1e-3),
        category="captioning",
    ),
)
