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
_CLASSIFICATION_EXPERIMENTS = [name for name in EXPERIMENTS if name != "captioning"]


@pytest.fixture(params=_CLASSIFICATION_EXPERIMENTS)
def classification_experiment_name(request: pytest.FixtureRequest) -> str:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def classification_experiment_cls(classification_experiment_name: str):
    cls, _ = EXPERIMENTS[classification_experiment_name]
    return cls
