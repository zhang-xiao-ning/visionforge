from training.evaluator import evaluate


def test_evaluate_returns_float(dummy_model, dummy_loader):
    acc = evaluate(dummy_model, dummy_loader)
    assert isinstance(acc, float)


def test_evaluate_in_range(dummy_model, dummy_loader):
    acc = evaluate(dummy_model, dummy_loader)
    assert 0.0 <= acc <= 1.0


def test_evaluate_restores_train_mode(dummy_model, dummy_loader):
    dummy_model.eval()
    evaluate(dummy_model, dummy_loader)
    assert dummy_model.training is True
