"""Contract: every classification model must satisfy these."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.datasets import DatasetInfo


def test_is_nn_module(classification_experiment_cls) -> None:
    model = classification_experiment_cls()
    assert isinstance(model, nn.Module)


def test_has_trainable_parameters(classification_experiment_cls) -> None:
    model = classification_experiment_cls()
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


def test_forward_output_shape(
    classification_experiment_cls,
    dataset_info: DatasetInfo,
) -> None:
    model = classification_experiment_cls()
    B = 2
    x = torch.randn(B, *dataset_info.input_shape)
    out = model(x)
    assert out.shape == (B, dataset_info.num_classes)


def test_backward_produces_grads(
    classification_experiment_cls,
    dataset_info: DatasetInfo,
) -> None:
    model = classification_experiment_cls()
    B = 2
    x = torch.randn(B, *dataset_info.input_shape)
    y = torch.randint(0, dataset_info.num_classes, (B,))

    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
