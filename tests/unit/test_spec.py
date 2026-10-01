"""Tests for spec dataclasses (TrainConfig / Stage / Experiment)."""

import dataclasses

import torch.nn as nn

from experiment.spec import Experiment, Stage, TrainConfig
from tasks.classification import ClassificationTask

# ---------- TrainConfig ----------


def test_train_config_defaults() -> None:
    cfg = TrainConfig()
    assert cfg.epochs == 1
    assert cfg.batch_size == 64
    assert cfg.optimizer == "sgd"


def test_train_config_has_no_experiment_field() -> None:
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    assert "experiment" not in fields


def test_train_config_has_no_seed_amp() -> None:
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    assert "seed" not in fields
    assert "amp" not in fields


# ---------- Stage ----------


def test_stage_defaults() -> None:
    s = Stage(name="align")
    assert s.name == "align"
    assert s.freeze == []
    assert s.overrides == {}


def test_stage_freeze_not_shared() -> None:
    a = Stage(name="a")
    b = Stage(name="b")
    a.freeze.append("vision")
    assert b.freeze == []


def test_stage_overrides_not_shared() -> None:
    a = Stage(name="a")
    b = Stage(name="b")
    a.overrides["epochs"] = 1
    assert b.overrides == {}


# ---------- Experiment ----------


class _Dummy(nn.Module):
    def forward(self, x):  # type: ignore[no-untyped-def]
        return x


def test_experiment_has_no_stages_field() -> None:
    fields = {f.name for f in dataclasses.fields(Experiment)}
    assert "stages" not in fields


def test_experiment_defaults() -> None:
    exp = Experiment(model=_Dummy, task=ClassificationTask, data="fake")
    assert exp.seed == 42
    assert exp.amp is False
    assert isinstance(exp.config, TrainConfig)
