from tasks.classification import ClassificationTask
from training.evaluator import evaluate


def test_evaluate_returns_metrics(dummy_model, dummy_loader):
    metrics = evaluate(dummy_model, dummy_loader, ClassificationTask())
    assert isinstance(metrics, dict)
    assert "acc" in metrics
    assert "loss" in metrics


def test_evaluate_accuracy_in_range(dummy_model, dummy_loader):
    metrics = evaluate(dummy_model, dummy_loader, ClassificationTask())
    assert 0.0 <= metrics["acc"] <= 1.0


def test_evaluate_restores_train_mode(dummy_model, dummy_loader):
    dummy_model.eval()
    evaluate(dummy_model, dummy_loader, ClassificationTask())
    assert dummy_model.training is True
