#!/usr/bin/env bash
# CI script: runs all checks locally or on any CI platform.
set -euo pipefail

echo "=========================================="
echo "CI: install dependencies"
echo "=========================================="
# 若在 CI 环境，强制使用 CPU 版 torch，避免下载 2.5GB 的 CUDA 库
if [ -n "${CI:-}" ]; then
    export UV_TORCH_BACKEND=cpu
    echo "(CI environment detected: using CPU torch)"
fi

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