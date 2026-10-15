#!/usr/bin/env python3
"""Dump source code into a single text file for review or archival.

Cross-platform: macOS / Linux / Windows.

Usage:
    uv run python scripts/dump_code.py                    # src/ + tests/ -> dump.txt
    uv run python scripts/dump_code.py --profile all      # whole project
    uv run python scripts/dump_code.py src/ scripts/      # custom paths
    uv run python scripts/dump_code.py --profile all --out snapshots/2026-09-29.txt

Profiles:
    default: src/ tests/
    all:     src/ tests/ scripts/ docs/ docker/ configs/ .github/
             + root config files
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT_DIR = Path("snapshots")

# Directories excluded at any depth (caches, VCS, build artifacts).
ALWAYS_EXCLUDE = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

# Directories skipped only when passed as a top-level path (data / training
# artifacts). A source subdirectory named "datasets" (e.g.
# src/application/datasets/) must still be dumped.
TOP_LEVEL_EXCLUDE = {"datasets", "checkpoints", "outputs", "exports"}

# File extensions that are collected.
WANTED_SUFFIXES = {
    ".py",
    ".pyi",
    ".toml",
    ".cfg",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".sh",
    ".sql",
}

# Files without an extension that are still collected.
WANTED_NAMES = {
    "Makefile",
    "Dockerfile",
    "LICENSE",
    "NOTICE",
    ".gitignore",
    ".dockerignore",
}

# Hard limit per file. Larger files are skipped and noted in the output.
MAX_FILE_BYTES = 1_000_000  # 1 MB

# Preset path sets. Entries with '*' are expanded via glob at runtime.
PROFILES: dict[str, list[str]] = {
    "default": ["src", "tests"],
    "all": [
        "src",
        "tests",
        "scripts",
        "docker",
        "configs",
        ".github",
        "pyproject.toml",
        "Makefile",
        "docker-compose.yml",
        "README.md",
        ".gitignore",
        ".dockerignore",
        ".pre-commit-config.yaml",
    ],
}

def _timestamp() -> str:
    """UTC timestamp safe for filenames: YYYYMMDD-HHMMSS."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _resolve_out_path(arg: str | None) -> Path:
    """Decide where to write.

    No --out  -> snapshots/dump-<UTC>.txt (never overwrites a previous dump).
    --out P   -> exactly P; caller is responsible for uniqueness.
    """
    if arg is None:
        return DEFAULT_OUT_DIR / f"dump-{_timestamp()}.txt"
    return Path(arg)

def is_wanted(p: Path) -> bool:
    if p.suffix in WANTED_SUFFIXES:
        return True
    name = p.name
    if name in WANTED_NAMES:
        return True
    if name.startswith("Dockerfile"):
        return True
    return False


def collect_files(paths: list[str]) -> list[Path]:
    """Expand paths (files or dirs) into a sorted list of files to dump."""
    files: list[Path] = []
    for p_str in paths:
        p = Path(p_str)
        if not p.exists():
            continue
        if p.is_file():
            files.append(p)
            continue
        if p.name in TOP_LEVEL_EXCLUDE:
            continue
        for dirpath, dirnames, filenames in os.walk(p):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if d not in ALWAYS_EXCLUDE and not d.endswith(".egg-info")
            )
            for fn in sorted(filenames):
                fp = Path(dirpath) / fn
                if is_wanted(fp):
                    files.append(fp)
    return sorted(set(files), key=lambda x: str(x))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="Files or directories to dump.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default=None,
        help="Named path set. Overrides positional paths.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output file. Default: snapshots/dump-<UTC timestamp>.txt "
            "(never overwrites a previous dump)."
        ),
    )
    args = parser.parse_args(argv[1:])

    if args.profile is not None:
        paths = PROFILES[args.profile]
    elif args.paths:
        paths = args.paths
    else:
        paths = PROFILES["default"]

    out_path = _resolve_out_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = collect_files(paths)

    with out_path.open("w", encoding="utf-8") as out:
        out.write("# Code dump\n")
        out.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
        out.write(f"# Profile:   {args.profile or '(custom)'}\n")
        out.write(f"# Paths:     {' '.join(paths)}\n")
        out.write(f"# Files:     {len(files)}\n")
        out.write("#\n\n")

        for f in files:
            size = f.stat().st_size
            if size > MAX_FILE_BYTES:
                out.write("=" * 64 + "\n")
                out.write(f"### FILE: {f}  (SKIPPED, {size} bytes > {MAX_FILE_BYTES})\n")
                out.write("=" * 64 + "\n\n")
                continue

            try:
                content = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                out.write("=" * 64 + "\n")
                out.write(f"### FILE: {f}  (SKIPPED, not UTF-8)\n")
                out.write("=" * 64 + "\n\n")
                continue

            lines = content.count("\n") + (
                0 if content.endswith("\n") or not content else 1
            )

            out.write("=" * 64 + "\n")
            out.write(f"### FILE: {f}  ({lines} lines)\n")
            out.write("=" * 64 + "\n\n")
            out.write(content)
            if content and not content.endswith("\n"):
                out.write("\n")
            out.write("\n\n")

    total_bytes = out_path.stat().st_size
    print(f"✅ Dumped {len(files)} files to {out_path} ({total_bytes:,} bytes)")
    print(f"   Profile: {args.profile or '(custom)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
