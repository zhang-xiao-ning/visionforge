"""Load experiment configuration from a YAML file.

YAML shape (only `experiment` is required):

    experiment: vit              # must exist in the registry
    description: baseline        # optional, human-readable
    learning_rate: 3e-4          # any TrainConfig field
    batch_size: 128
    stages:                      # optional, list of stages
      - name: align
        freeze: [vision]
        epochs: 1

Any top-level key not in {experiment, description, stages} must be a
valid TrainConfig field name. Unknown keys raise ValueError.

This module does NOT merge — merging happens in `cli.build_run` so that
the full priority chain (TrainConfig default < Experiment.config < YAML
< CLI < stage.overrides) is in one place.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from framework.experiment.spec import Stage, TrainConfig

_RESERVED_KEYS = {"experiment", "description", "stages"}


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and validate the top-level shape."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(data).__name__}")

    if "experiment" not in data:
        raise ValueError("YAML must contain an 'experiment' key")

    if not isinstance(data["experiment"], str):
        raise ValueError("'experiment' must be a string")

    return data


def split_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Return the top-level keys that are meant as TrainConfig overrides."""
    return {k: v for k, v in data.items() if k not in _RESERVED_KEYS}


def validate_override_keys(overrides: dict[str, Any]) -> None:
    """Check that every key in `overrides` is a valid TrainConfig field."""
    valid = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"Unknown config keys: {sorted(unknown)}. Valid TrainConfig fields: {sorted(valid)}"
        )


def coerce_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Cast override values to match TrainConfig field types.

    YAML 1.1 does not treat `1e-4` as a float (only `1.0e-4` or `0.0001`
    are). This function casts each value to its declared field type so
    users can write natural Python-style literals.
    """
    hints = get_type_hints(TrainConfig)
    coerced: dict[str, Any] = {}
    for name, value in overrides.items():
        target = hints.get(name)
        if target in (int, float, str, bool):
            try:
                coerced[name] = target(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Cannot cast {name}={value!r} to {target.__name__}: {e}") from e
        else:
            coerced[name] = value
    return coerced


def parse_stages(raw: list[Any]) -> list[Stage]:
    """Parse a list of stage dicts (from YAML) into Stage objects."""
    if not isinstance(raw, list):
        raise ValueError("'stages' must be a list")

    stages: list[Stage] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"stages[{i}] must be a mapping")
        if "name" not in item:
            raise ValueError(f"stages[{i}] must have a 'name' key")
        if not isinstance(item["name"], str):
            raise ValueError(f"stages[{i}].name must be a string")

        freeze = item.get("freeze", [])
        if not isinstance(freeze, list):
            raise ValueError(f"stages[{i}].freeze must be a list")

        overrides = {k: v for k, v in item.items() if k not in {"name", "freeze"}}
        validate_override_keys(overrides)
        overrides = coerce_overrides(overrides)

        stages.append(Stage(name=item["name"], freeze=freeze, overrides=overrides))
    return stages
