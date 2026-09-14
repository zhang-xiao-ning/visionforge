import torch

from config import device, dtype


def evaluate(model, loader):
    """返回 accuracy (float, 0~1)。"""
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum().item()
            num_samples += preds.size(0)

    model.train()
    return num_correct / num_samples