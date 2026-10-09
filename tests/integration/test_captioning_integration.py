"""Integration tests for captioning with real Flickr8k data."""

import pytest

from data.bundle import DataContext
from data.datasets import build_data
from models.captioning import CaptioningModel
from training.strategy import SingleDeviceStrategy

pytestmark = pytest.mark.integration


def test_captioning_bundle_and_model(require_flickr8k) -> None:
    """build_data(flickr8k) + CaptioningModel.from_data work together."""
    bundle, eval_bundle = build_data(
        "flickr8k",
        DataContext(batch_size=4, strategy=SingleDeviceStrategy(), num_train=2),
    )
    model = CaptioningModel.from_data(bundle)

    assert model.lm_head.out_features == bundle.model_init["vocab_size"]
    assert "tokenizer_name" in bundle.extras
    assert eval_bundle is not None  # flickr8k ships an image-level eval set
