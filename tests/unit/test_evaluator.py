from evaluation import Accuracy, CrossEntropy
from training.evaluator import evaluate


def test_evaluate_returns_metrics(dummy_model, dummy_loader):
    metrics = evaluate(dummy_model, dummy_loader, [Accuracy(), CrossEntropy()])
    assert isinstance(metrics, dict)
    assert "acc" in metrics
    assert "loss" in metrics


def test_evaluate_accuracy_in_range(dummy_model, dummy_loader):
    metrics = evaluate(dummy_model, dummy_loader, [Accuracy()])
    assert 0.0 <= metrics["acc"] <= 1.0


def test_evaluate_restores_train_mode(dummy_model, dummy_loader):
    dummy_model.eval()
    evaluate(dummy_model, dummy_loader, [Accuracy()])
    assert dummy_model.training is True


def test_evaluate_empty_metrics_returns_empty_dict(dummy_model, dummy_loader):
    assert evaluate(dummy_model, dummy_loader, []) == {}
