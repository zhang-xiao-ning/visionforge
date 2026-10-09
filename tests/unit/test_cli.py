"""Tests for CLI + YAML merging logic."""

from argparse import Namespace
from pathlib import Path

import pytest

from cli import build_run


def _ns(**kwargs) -> Namespace:
    """Namespace with all CLI defaults, overridable via kwargs."""
    defaults: dict = {
        "config": None,
        "experiment": None,
        "seed": None,
        "amp": False,
        "num_train": None,
        "resume": None,
        # TrainConfig fields
        "epochs": None,
        "batch_size": None,
        "optimizer": None,
        "learning_rate": None,
        "momentum": None,
        "no_nesterov": False,
        "weight_decay": None,
        "lr_scheduler": None,
        "step_size": None,
        "gamma": None,
        "warmup_steps": None,
        "accum_steps": None,
        "grad_clip": None,
        "early_stop_patience": None,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


def _write_yaml(tmp_path: Path, content: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(content)
    return str(p)


# ---------- no YAML ----------


def test_default_experiment() -> None:
    params = build_run(_ns())
    assert params.experiment_name == "mlp"  # DEFAULT_EXPERIMENT
    assert params.stages is None


def test_explicit_experiment_and_cli_override() -> None:
    params = build_run(_ns(experiment="vit", epochs=3))
    assert params.experiment_name == "vit"
    assert params.config.epochs == 3


# ---------- with YAML ----------


def test_yaml_sets_experiment_and_config(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\nepochs: 5\nlearning_rate: 1e-4\n")
    params = build_run(_ns(config=path))
    assert params.experiment_name == "vit"
    assert params.config.epochs == 5
    assert params.config.learning_rate == 1e-4
    assert params.stages is None


def test_cli_overrides_yaml(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\nepochs: 5\n")
    params = build_run(_ns(config=path, epochs=10))
    assert params.config.epochs == 10


def test_experiment_conflict_raises(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\n")
    with pytest.raises(ValueError, match="conflicts"):
        build_run(_ns(config=path, experiment="captioning"))


def test_unregistered_experiment_in_yaml_raises(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: nope\n")
    with pytest.raises(ValueError, match="not registered"):
        build_run(_ns(config=path))


def test_unknown_yaml_key_raises(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\nnope: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        build_run(_ns(config=path))


# ---------- stages ----------


def test_yaml_stages_parsed(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
experiment: vit
stages:
  - name: a
    freeze: [x]
    epochs: 1
  - name: b
    epochs: 2
""",
    )
    params = build_run(_ns(config=path))
    assert params.stages is not None
    assert len(params.stages) == 2
    assert params.stages[0].name == "a"
    assert params.stages[0].freeze == ["x"]
    assert params.stages[0].overrides == {"epochs": 1}
    assert params.stages[1].name == "b"
    assert params.stages[1].freeze == []
    assert params.stages[1].overrides == {"epochs": 2}


def test_no_stages_field_means_single_stage(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\n")
    params = build_run(_ns(config=path))
    assert params.stages is None


# ---------- seed / amp ----------


def test_seed_defaults_to_experiment(tmp_path: Path) -> None:
    params = build_run(_ns(experiment="vit"))
    assert params.seed == 42


def test_cli_seed_overrides(tmp_path: Path) -> None:
    params = build_run(_ns(experiment="vit", seed=123))
    assert params.seed == 123


def test_amp_or_logic(tmp_path: Path) -> None:
    params = build_run(_ns(experiment="vit", amp=True))
    assert params.amp is True
