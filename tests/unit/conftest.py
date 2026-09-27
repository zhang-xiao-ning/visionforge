"""All tests under tests/unit/ are unit tests."""

import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark every test in this directory as unit."""
    for item in items:
        item.add_marker(pytest.mark.unit)
