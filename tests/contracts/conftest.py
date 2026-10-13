"""Fixtures for contract tests."""

import pytest

from framework.registry import EXPERIMENTS

_CLASSIFICATION_EXPERIMENTS = [
    name for name, exp in EXPERIMENTS.items() if exp.category == "classification"
]


@pytest.fixture(params=_CLASSIFICATION_EXPERIMENTS)
def classification_experiment_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def classification_experiment_cls(classification_experiment_name: str):
    return EXPERIMENTS[classification_experiment_name].model
