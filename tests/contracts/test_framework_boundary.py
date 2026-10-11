"""Contract: framework must not import application code."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent.parent / "src"
FRAMEWORK = SRC / "framework"

FORBIDDEN = {
    "application",
    "data",
    "datasets",
    "models",
    "tasks",
    "metrics",
    "serving",
    "export",
    "registry",
    "cli",
    "main",
}


def _iter_py(root: Path):
    yield from root.rglob("*.py")


def _imports(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
    return names


@pytest.mark.parametrize(
    "path",
    sorted(_iter_py(FRAMEWORK)),
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_framework_does_not_import_application(path: Path) -> None:
    violations = _imports(path.read_text()) & FORBIDDEN
    assert not violations, f"{path.relative_to(SRC)} imports application: {violations}"
