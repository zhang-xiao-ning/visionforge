"""Tests for CLI + YAML merging logic.

`build_run()` reads `sys.argv` internally. Tests drive it by
monkeypatching `sys.argv` (via the `run_cli` fixture) rather than
refactoring the production signature — argparse then runs the real
parse path, so choices / type coercion / mutual exclusion are all
covered.
"""

from pathlib import Path

import pytest

import application.registry  # noqa: F401 — populates EXPERIMENTS / DATASETS
from framework.cli import build_run

# ---------- helpers ----------


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch):
    """Call `build_run()` with a synthetic argv."""

    def _run(*argv: str):
        monkeypatch.setattr("sys.argv", ["visionforge", *argv])
        return build_run()

    return _run


def _write_yaml(tmp_path: Path, content: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(content)
    return str(p)


# ---------- no YAML ----------


def test_default_experiment(run_cli) -> None:
    params = run_cli()
    assert params.experiment_name == "mlp"  # DEFAULT_EXPERIMENT
    assert params.stages is None


def test_explicit_experiment(run_cli) -> None:
    params = run_cli("--experiment", "vit")
    assert params.experiment_name == "vit"


def test_cli_override_takes_effect(run_cli) -> None:
    params = run_cli("--experiment", "vit", "--epochs", "3", "--batch-size", "32")
    assert params.config.epochs == 3
    assert params.config.batch_size == 32


def test_cli_multi_flag_override(run_cli) -> None:
    params = run_cli(
        "--experiment",
        "captioning",
        "--optimizer",
        "adamw",
        "--learning-rate",
        "1e-4",
        "--weight-decay",
        "0.01",
        "--warmup-steps",
        "100",
        "--accum-steps",
        "4",
        "--grad-clip",
        "1.0",
    )
    assert params.config.optimizer == "adamw"
    assert params.config.learning_rate == 1e-4
    assert params.config.weight_decay == 0.01
    assert params.config.warmup_steps == 100
    assert params.config.accum_steps == 4
    assert params.config.grad_clip == 1.0


def test_lr_scheduler_const_is_accepted(run_cli) -> None:
    """Regression: 'const' must be a valid choice, since it's the default."""
    params = run_cli("--lr-scheduler", "const")
    assert params.config.lr_scheduler == "const"


def test_no_nesterov_flag(run_cli) -> None:
    params = run_cli("--experiment", "vit", "--no-nesterov")
    assert params.config.nesterov is False


# ---------- with YAML ----------


def test_yaml_sets_experiment_and_config(run_cli, tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\nepochs: 5\nlearning_rate: 1e-4\n")
    params = run_cli("--config", path)
    assert params.experiment_name == "vit"
    assert params.config.epochs == 5
    assert params.config.learning_rate == 1e-4
    assert params.stages is None


def test_cli_overrides_yaml(run_cli, tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\nepochs: 5\n")
    params = run_cli("--config", path, "--epochs", "10")
    assert params.config.epochs == 10


def test_experiment_conflict_raises(run_cli, tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\n")
    with pytest.raises(ValueError, match="conflicts"):
        run_cli("--config", path, "--experiment", "captioning")


def test_unregistered_experiment_in_yaml_raises(run_cli, tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: nope\n")
    with pytest.raises(ValueError, match="not registered"):
        run_cli("--config", path)


def test_unknown_yaml_key_raises(run_cli, tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "experiment: vit\nnope: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        run_cli("--config", path)


# ---------- stages ----------


def test_yaml_stages_parsed(run_cli, tmp_path: Path) -> None:
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
    params = run_cli("--config", path)
    assert params.stages is not None
    assert len(params.stages) == 2
    assert params.stages[0].name == "a"
    assert params.stages[0].freeze == ["x"]
    assert params.stages[0].overrides == {"epochs": 1}
    assert params.stages[1].name == "b"
    assert params.stages[1].freeze == []
    assert params.stages[1].overrides == {"epochs": 2}


# ---------- seed / amp / misc globals ----------


def test_seed_defaults_to_experiment(run_cli) -> None:
    params = run_cli("--experiment", "vit")
    assert params.seed == 42


def test_cli_seed_overrides(run_cli) -> None:
    params = run_cli("--experiment", "vit", "--seed", "123")
    assert params.seed == 123


def test_amp_flag(run_cli) -> None:
    params = run_cli("--experiment", "vit", "--amp")
    assert params.amp is True


def test_num_train_passthrough(run_cli) -> None:
    params = run_cli("--experiment", "vit", "--num-train", "1000")
    assert params.num_train == 1000


def test_resume_passthrough(run_cli) -> None:
    params = run_cli("--experiment", "vit", "--resume", "checkpoints/x.pt")
    assert params.resume == "checkpoints/x.pt"
