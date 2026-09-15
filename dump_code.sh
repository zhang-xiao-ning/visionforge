#!/usr/bin/env bash
set -euo pipefail

OUTPUT="all_code.txt"
> "$OUTPUT"

find . -type f \( \
    -name "*.py" -o \
    -name "*.sh" -o \
    -name "*.md" -o \
    -name "*.txt" -o \
    -name "*.toml" -o \
    -name "*.yaml" -o \
    -name ".gitignore" -o \
    -name "*.typed" -o \
    -name "*.yml" \
  \) \
  -not -path "./.venv/*" \
  -not -path "./datasets/*" \
  -not -path "./.git/*" \
  -not -path "./.idea/*" \
  -not -path "*/__pycache__/*" \
  -not -path "./outputs/*" \
  -not -path "./checkpoints/*" \
  -not -path "./*.egg-info/*" \
  -not -path "*.egg-info/*" \
  -not -path "./all_code.txt" \
  | sort | while read -r f; do
    echo "===== $f =====" >> "$OUTPUT"
    cat "$f" >> "$OUTPUT"
    echo -e "\n\n" >> "$OUTPUT"
done

echo "已生成 $OUTPUT"