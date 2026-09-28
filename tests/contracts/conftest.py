"""Fixtures for contract tests.

Contracts define what *every* implementation must satisfy, so they run
against all registered experiments / datasets / tokenizers automatically.

To add a new implementation, register it and update the `params` list
below. No test code changes.
"""

import pytest

from registry import EXPERIMENTS

# Classification experiments eligible for the classification contract.
# Captioning is excluded because its forward() has a different signature.
_CLASSIFICATION_EXPERIMENTS = [
    name for name, exp in EXPERIMENTS.items() if exp["category"] == "classification"
]


@pytest.fixture(params=_CLASSIFICATION_EXPERIMENTS)
def classification_experiment_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def classification_experiment_cls(classification_experiment_name: str):
    return EXPERIMENTS[classification_experiment_name]["model"]
