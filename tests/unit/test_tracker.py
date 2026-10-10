"""Tests for MetricTracker."""

import torch.nn as nn

from framework.training import MetricTracker


def _model() -> nn.Module:
    return nn.Linear(4, 2)


def test_higher_is_better_starts_at_neg_inf():
    t = MetricTracker(higher_is_better=True)
    assert t.best == float("-inf")


def test_lower_is_better_starts_at_pos_inf():
    t = MetricTracker(higher_is_better=False)
    assert t.best == float("inf")


def test_initial_best_overrides_default():
    t = MetricTracker(higher_is_better=True, initial_best=0.5)
    assert t.best == 0.5


def test_update_improves_snapshots():
    t = MetricTracker(higher_is_better=True)
    t.update(0.5, _model())
    assert t.best == 0.5
    assert t.best_state is not None
    assert t.epochs_no_improve == 0


def test_update_no_improvement_increments():
    t = MetricTracker(higher_is_better=True)
    m = _model()
    t.update(0.5, m)
    assert t.update(0.3, m) is True  # patience=0 → never stop
    assert t.epochs_no_improve == 1


def test_early_stop_after_patience():
    t = MetricTracker(higher_is_better=True, patience=2)
    m = _model()
    t.update(0.5, m)
    assert t.update(0.3, m) is True
    assert t.update(0.2, m) is False  # 2 consecutive no-improve


def test_lower_is_better_improvement():
    t = MetricTracker(higher_is_better=False)
    m = _model()
    t.update(0.5, m)
    assert t.update(0.3, m) is True
    assert t.best == 0.3


def test_restore_is_noop_without_snapshot():
    t = MetricTracker(higher_is_better=True)
    t.restore(_model())  # should not raise


def test_restore_after_snapshot():
    m = _model()
    t = MetricTracker(higher_is_better=True)
    t.update(0.5, m)
    # Mutate weights, then restore
    for p in m.parameters():
        p.data.fill_(0.0)
    t.restore(m)
    # Weights should now be from the snapshot (not all zeros)
    assert any(p.abs().sum().item() > 0 for p in m.parameters())
