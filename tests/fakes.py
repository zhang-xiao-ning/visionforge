"""Shared fakes for unit tests.

Not a pytest fixture module — imported as regular functions/classes.
Rationale: most ML test fakes don't need setup/teardown, so a plain
function is simpler and more flexible than a fixture.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from framework.interfaces import DataBundle, Model, Task


def make_classification_builder(
    *,
    shape: tuple[int, ...] = (3, 32, 32),
    num_classes: int = 10,
    n: int = 8,
    batch_size: int = 4,
    model_init: dict | None = None,
):
    """Return a builder(ctx) -> (DataBundle, None) for runner tests.

    model_init: kwargs passed to `Model.from_data`. Default
                {"num_classes": num_classes}. Pass {} for models that
                take no constructor args (e.g. _TwoGroupModel).
    """
    if model_init is None:
        model_init = {"num_classes": num_classes}

    def build(ctx) -> tuple[DataBundle, None]:  # noqa: ARG001
        x = torch.randn(n, *shape)
        y = torch.randint(0, num_classes, (n,))
        loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
        return DataBundle(
            loader_train=loader,
            loader_val=loader,
            loader_test=loader,
            model_init=model_init,
            extras={},
        ), None

    return build


class TinyModel(Model):
    """4-dim input → N-class output. Minimal Model for runner tests."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TinyTask(Task):
    def train_step(self, model, batch, device, dtype):
        x, y = batch
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)
        return nn.functional.cross_entropy(model(x), y)
