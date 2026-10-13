from application.datasets import cifar10, flickr8k, tinyshakespeare
from application.metrics import BLEU4
from application.models import MLP, CaptioningModel, MambaClassifier, NanoGPT, ViT
from application.tasks import CaptioningTask, ClassificationTask, LMTask
from framework.experiment import Experiment, TrainConfig
from framework.metrics import Accuracy, CrossEntropy, Perplexity, TokenPerplexity
from framework.registry import DATASETS, EXPERIMENTS

# Each entry is written out in full, even though the classification
# entries share most fields. This is deliberate: the registry is a
# configuration file, and a small amount of duplication keeps each
# experiment self-contained and independently editable. Extracting a
# helper would force every future divergence (a different metric, a
# different dataset) to flow through the helper's signature.
DATASETS.register("cifar10", cifar10.build_bundle)
DATASETS.register("flickr8k", flickr8k.build_bundle)
DATASETS.register("tinyshakespeare", tinyshakespeare.build_bundle)

EXPERIMENTS.register(
    Experiment(
        name="mlp",
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
    Experiment(
        name="vit",
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
    Experiment(
        name="mamba",
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
    Experiment(
        name="captioning",
        model=CaptioningModel,
        task=CaptioningTask,
        data="flickr8k",
        metrics=[Perplexity, BLEU4],
        primary_metric="perplexity",
        config=TrainConfig(learning_rate=1e-3),
        category="captioning",
    ),
)
EXPERIMENTS.register(
    Experiment(
        name="nanogpt",
        model=NanoGPT,
        task=LMTask,
        data="tinyshakespeare",
        metrics=[TokenPerplexity],
        primary_metric="perplexity",
        config=TrainConfig(
            optimizer="adamw",
            learning_rate=3e-4,
            batch_size=64,
            weight_decay=0.1,
            warmup_steps=100,
            lr_scheduler="cosine",
        ),
        category="language_modeling",
    ),
)
