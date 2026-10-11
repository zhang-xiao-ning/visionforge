"""Contract: application only uses post-refactor import paths.

The framework/application split removed several top-level packages
from src/ (data, models, tasks, training, utils, experiment,
interfaces, runtime, registry, cli, main). Any import of those names
inside application/ means a file wasn't updated during the split.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent.parent / "src"
APPLICATION = SRC / "application"

STALE_TOP_LEVEL = frozenset(
    {
        "data",
        "runtime",
        "utils",
        "training",
        "models",
        "tasks",
        "evaluation",
        "experiment",
        "interfaces",
        "registry",
        "cli",
        "main",
    }
)


def _iter_py(root: Path):
    yield from root.rglob("*.py")


def _stale_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            top = node.module.split(".")[0]
            if top in STALE_TOP_LEVEL:
                bad.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in STALE_TOP_LEVEL:
                    bad.append(f"import {alias.name}")
    return bad


@pytest.mark.parametrize(
    "path",
    sorted(_iter_py(APPLICATION)),
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_application_uses_post_refactor_imports(path: Path) -> None:
    bad = _stale_imports(path.read_text())
    assert not bad, f"{path.relative_to(SRC)} uses stale import paths: {bad}"
