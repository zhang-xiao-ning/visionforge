#!/usr/bin/env bash
# CI script: runs all checks locally or on any CI platform.
set -euo pipefail

echo "=========================================="
echo "CI: install dependencies"
echo "=========================================="
uv sync

echo ""
echo "=========================================="
echo "CI: ruff check"
echo "=========================================="
uv run ruff check src/ tests/

echo ""
echo "=========================================="
echo "CI: ruff format --check"
echo "=========================================="
uv run ruff format --check src/ tests/

echo ""
echo "=========================================="
echo "CI: mypy"
echo "=========================================="
uv run mypy src/

echo ""
echo "=========================================="
echo "CI: pytest"
echo "=========================================="
uv run pytest -v

echo ""
echo "=========================================="
echo "CI: all checks passed"
echo "=========================================="