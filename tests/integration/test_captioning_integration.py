"""Integration tests for captioning setup with real Flickr8k data."""

import pytest

from experiment.config import TrainConfig
from models.base import SetupContext
from models.captioning import CaptioningModel

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="requires Flickr8k + network (tiktoken); run manually")
def test_setup_returns_bundle() -> None:
    """setup() returns (model, task, loaders) with consistent vocab."""
    cfg = TrainConfig(experiment="captioning", epochs=1)
    ctx = SetupContext(
        cfg=cfg,
        dataset_name="flickr8k",
        batch_size=8,
        num_train=10,
    )
    bundle = CaptioningModel.setup(ctx)

    assert isinstance(bundle.model, CaptioningModel)
    assert bundle.task is not None
    assert bundle.loader_train is not None
