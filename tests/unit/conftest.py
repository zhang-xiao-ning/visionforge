"""All tests under tests/unit/ are unit tests."""

from collections.abc import Iterator

import pytest

from framework.utils import AppLogger


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-mark every test in this directory as unit."""
    for item in items:
        item.add_marker(pytest.mark.unit)


@pytest.fixture
def dummy_logger() -> Iterator[AppLogger]:
    """Minimal AppLogger for tests that just need the interface filled in.

    Writes nothing to disk and only emits ERROR+ to stderr, so test
    output stays clean.
    """
    logger = AppLogger("test", level="error")
    try:
        yield logger
    finally:
        logger.close()
