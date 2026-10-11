"""Tests for framework.utils.path."""

from framework.utils.path import CHECKPOINTS_PATH, DATASETS_PATH, OUTPUTS_PATH, PROJECT_ROOT


def test_project_root_contains_pyproject() -> None:
    """PROJECT_ROOT must point at the actual project root, not src/."""
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_output_paths_are_under_project_root() -> None:
    for path in (DATASETS_PATH, CHECKPOINTS_PATH, OUTPUTS_PATH):
        assert path.parent == PROJECT_ROOT


def test_datasets_path_is_where_readme_says() -> None:
    """README documents datasets/ at the repo root."""
    assert DATASETS_PATH.name == "datasets"
    assert (PROJECT_ROOT / "datasets") == DATASETS_PATH
