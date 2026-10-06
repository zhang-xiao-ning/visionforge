"""Registry of available experiments."""

from evaluation import BLEU4, Accuracy, CrossEntropy, Perplexity
from experiment.spec import Experiment, TrainConfig
from models.captioning import CaptioningModel
from models.mamba import MambaClassifier
from models.mlp import MLP
from models.vit import ViT
from tasks.captioning import CaptioningTask
from tasks.classification import ClassificationTask

EXPERIMENTS: dict[str, Experiment] = {
    "mlp": Experiment(
        model=MLP,
        task=ClassificationTask,
        data="cifar10",
        metrics=[Accuracy, CrossEntropy],
        primary_metric="acc",
        config=TrainConfig(learning_rate=1e-2),
        category="classification",
    ),
    "vit": Experiment(
        model=ViT,
        task=ClassificationTask,
        data="cifar10",
        metrics=[Accuracy, CrossEntropy],
        primary_metric="acc",
        config=TrainConfig(learning_rate=3e-4),
        category="classification",
    ),
    "mamba": Experiment(
        model=MambaClassifier,
        task=ClassificationTask,
        data="cifar10",
        metrics=[Accuracy, CrossEntropy],
        primary_metric="acc",
        config=TrainConfig(learning_rate=3e-4, batch_size=64),
        category="classification",
    ),
    "captioning": Experiment(
        model=CaptioningModel,
        task=CaptioningTask,
        data="flickr8k",
        metrics=[Perplexity, BLEU4],
        primary_metric="perplexity",
        config=TrainConfig(learning_rate=1e-3),
        category="captioning",
    ),
}
