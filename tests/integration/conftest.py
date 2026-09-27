"""Shared fixtures for integration tests.

Integration tests need real datasets + environment flags. These fixtures
explicitly declare which dataset each test requires. No autouse — that
would incorrectly skip tests unrelated to the missing data.
"""

import os

import pytest

from utils.path import DATASETS_PATH

CIFAR10_PATH = DATASETS_PATH / "cifar-10-batches-py"
FLICKR8K_PATH = DATASETS_PATH / "Flicker8k_Dataset"

_RUN_INTEGRATION = os.environ.get("RUN_INTEGRATION", "").lower() in (
    "1",
    "true",
    "yes",
)
_RUN_REGRESSION = os.environ.get("RUN_REGRESSION", "").lower() in (
    "1",
    "true",
    "yes",
)


def _has_cifar10() -> bool:
    return (CIFAR10_PATH / "test_batch").exists()


def _has_flickr8k() -> bool:
    return FLICKR8K_PATH.exists()


@pytest.fixture
def require_integration() -> None:
    """Skip unless RUN_INTEGRATION=1."""
    if not _RUN_INTEGRATION:
        pytest.skip("set RUN_INTEGRATION=1 to run")


@pytest.fixture
def require_regression() -> None:
    """Skip unless RUN_REGRESSION=1."""
    if not _RUN_REGRESSION:
        pytest.skip("set RUN_REGRESSION=1 to run")


@pytest.fixture
def require_cifar10() -> None:
    """Skip if CIFAR-10 data is not on disk."""
    if not _has_cifar10():
        pytest.skip(f"CIFAR-10 not found at {CIFAR10_PATH}")


@pytest.fixture
def require_flickr8k() -> None:
    """Skip if Flickr8k data is not on disk."""
    if not _has_flickr8k():
        pytest.skip(f"Flickr8k not found at {FLICKR8K_PATH}")
