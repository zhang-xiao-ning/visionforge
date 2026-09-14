#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME=$(basename "$PWD")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${PROJECT_NAME}_${TIMESTAMP}.zip"

echo "Zip to $OUTPUT ..."

zip -r "$OUTPUT" . \
  -x ".venv/*" \
  -x ".venv" \
  -x "datasets/*" \
  -x "checkpoints/*" \
  -x "outputs/*" \
  -x ".git/*" \
  -x ".idea/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*.pyo" \
  -x "*.zip" \
  -x "*.pt" \
  -x "*.pth" \
  -x "*.ckpt" \
  -x "*.DS_Store" \
  -x ".env" \
  -x ".env.*" \
  -x "*.log" \
  -x ".mypy_cache/*" \
  -x ".pytest_cache/*" \
  -x ".ruff_cache/*" \
  -x ".ipynb_checkpoints/*" \
  -x "$OUTPUT"

echo "Complete: $OUTPUT"