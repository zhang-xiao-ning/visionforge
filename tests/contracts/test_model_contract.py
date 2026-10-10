"""Contract: every registered model inherits Model and satisfies its interface."""

import pytest

from application.registry import EXPERIMENTS
from framework.interfaces import Model


@pytest.fixture(params=list(EXPERIMENTS.keys()))
def model_cls(request: pytest.FixtureRequest):
    return EXPERIMENTS[request.param].model


def test_inherits_model_base(model_cls) -> None:
    assert issubclass(model_cls, Model)


def test_param_groups_default_returns_dict_of_lists(classification_experiment_cls) -> None:
    model = classification_experiment_cls()
    groups = model.param_groups()
    assert isinstance(groups, dict)
    assert len(groups) >= 1
    for name, params in groups.items():
        assert isinstance(name, str)
        assert isinstance(params, list)
        assert all(hasattr(p, "requires_grad") for p in params)


def test_param_groups_default_covers_all_parameters(
    classification_experiment_cls,
) -> None:
    model = classification_experiment_cls()
    groups = model.param_groups()
    all_params = [p for ps in groups.values() for p in ps]
    assert set(map(id, all_params)) == set(map(id, model.parameters()))


def test_initialize_default_is_noop(
    classification_experiment_cls,
    tmp_path,
) -> None:
    model = classification_experiment_cls()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.initialize(tmp_path / "nonexistent.pt")
    after = model.state_dict()
    assert set(before.keys()) == set(after.keys())
    for k in before:
        assert (before[k] == after[k]).all()
