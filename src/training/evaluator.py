import torch

from config import device, dtype


def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on validation set")
    else:
        print("Checking accuracy on test set")

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum().item()
            num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print("Got %d / %d correct (%.2f)" % (num_correct, num_samples, 100 * acc))