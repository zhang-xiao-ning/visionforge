"""Tests for the package version."""

import re

import src


def test_version_is_string() -> None:
    assert isinstance(src.__version__, str)


def test_version_matches_pattern() -> None:
    # 允许：0.1.0 / 0.1.1.dev3 / 0.0.0+unknown
    pattern = r"^\d+\.\d+\.\d+([-.+][A-Za-z0-9.]+)?$"
    assert re.match(pattern, src.__version__), f"Unexpected version format: {src.__version__}"


def test_version_not_unknown_in_installed_env() -> None:
    # 在 CI / uv sync 环境下，应该是真实版本，不是 fallback
    assert src.__version__ != "0.0.0+unknown"
