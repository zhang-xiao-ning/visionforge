"""Tests for ONNX export spec resolution.

Does not run the actual export — that needs onnxruntime and a real
checkpoint. Only verifies that the dummy inputs and I/O metadata are
derived from the experiment's declared dataset, not from literals.
"""

import pytest
import torch

from application.export.onnx_export import _export_spec
from application.registry import EXPERIMENTS


def test_classification_uses_dataset_input_shape() -> None:
    dummy_args, io = _export_spec(EXPERIMENTS["vit"])
    assert len(dummy_args) == 1
    assert dummy_args[0].shape == (1, 3, 32, 32)
    assert io["input_names"] == ["input"]
    assert io["output_names"] == ["output"]


def test_captioning_has_two_inputs() -> None:
    dummy_args, io = _export_spec(EXPERIMENTS["captioning"])
    images, ids = dummy_args
    assert images.shape == (1, 3, 224, 224)
    assert ids.dtype == torch.long
    assert io["input_names"] == ["images", "input_ids"]


def test_language_modeling_raises() -> None:
    with pytest.raises(ValueError, match="language_modeling"):
        _export_spec(EXPERIMENTS["nanogpt"])
