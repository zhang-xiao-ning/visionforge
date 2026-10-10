"""Tests for YAML experiment loader."""

from pathlib import Path

import pytest

from framework.experiment.loader import (
    coerce_overrides,
    load_yaml,
    parse_stages,
    split_overrides,
    validate_override_keys,
)


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(content)
    return p


# ---------- load_yaml ----------


def test_load_yaml_minimal(tmp_path: Path) -> None:
    p = _write(tmp_path, "experiment: vit\n")
    data = load_yaml(p)
    assert data["experiment"] == "vit"


def test_load_yaml_missing_experiment(tmp_path: Path) -> None:
    p = _write(tmp_path, "epochs: 5\n")
    with pytest.raises(ValueError, match="experiment"):
        load_yaml(p)


def test_load_yaml_root_not_mapping(tmp_path: Path) -> None:
    p = _write(tmp_path, "- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(p)


# ---------- split_overrides ----------


def test_split_overrides_strips_reserved() -> None:
    data = {"experiment": "vit", "description": "x", "stages": [], "epochs": 5}
    assert split_overrides(data) == {"epochs": 5}


def test_split_overrides_empty_when_only_reserved() -> None:
    assert split_overrides({"experiment": "vit"}) == {}


# ---------- validate_override_keys ----------


def test_validate_override_keys_accepts_known() -> None:
    validate_override_keys({"epochs": 5, "learning_rate": 1e-3})


def test_validate_override_keys_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown config keys"):
        validate_override_keys({"nope": 1})


# ---------- parse_stages ----------


def test_parse_stages_basic() -> None:
    stages = parse_stages([{"name": "a", "freeze": ["x"], "epochs": 1}])
    assert len(stages) == 1
    assert stages[0].name == "a"
    assert stages[0].freeze == ["x"]
    assert stages[0].overrides == {"epochs": 1}


def test_parse_stages_coerces_overrides() -> None:
    stages = parse_stages([{"name": "a", "learning_rate": "1e-4"}])
    assert stages[0].overrides == {"learning_rate": 1e-4}
    assert isinstance(stages[0].overrides["learning_rate"], float)


def test_parse_stages_not_list() -> None:
    with pytest.raises(ValueError, match="list"):
        parse_stages({"name": "a"})  # type: ignore[arg-type]


def test_parse_stages_item_not_mapping() -> None:
    with pytest.raises(ValueError, match="mapping"):
        parse_stages(["just a string"])  # type: ignore[list-item]


def test_parse_stages_missing_name() -> None:
    with pytest.raises(ValueError, match="name"):
        parse_stages([{"epochs": 1}])


def test_parse_stages_freeze_not_list() -> None:
    with pytest.raises(ValueError, match="freeze"):
        parse_stages([{"name": "a", "freeze": "x"}])


def test_parse_stages_unknown_override() -> None:
    with pytest.raises(ValueError, match="Unknown config keys"):
        parse_stages([{"name": "a", "nope": 1}])


# ---------- coerce_overrides ----------


def test_coerce_string_to_float() -> None:
    """YAML 1.1 parses `1e-4` as a string; we cast to float."""
    result = coerce_overrides({"learning_rate": "1e-4", "weight_decay": "0.01"})
    assert result == {"learning_rate": 1e-4, "weight_decay": 0.01}
    assert isinstance(result["learning_rate"], float)


def test_coerce_string_to_int() -> None:
    result = coerce_overrides({"epochs": "5"})
    assert result == {"epochs": 5}
    assert isinstance(result["epochs"], int)


def test_coerce_float_to_int() -> None:
    result = coerce_overrides({"epochs": 5.0})
    assert result == {"epochs": 5}
    assert isinstance(result["epochs"], int)


def test_coerce_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Cannot cast"):
        coerce_overrides({"epochs": "not-a-number"})
