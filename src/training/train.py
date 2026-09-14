import torch
import torch.nn.functional as F

from config import device, dtype, print_every
from training.evaluator import check_accuracy_part34


def train_part34(model, optimizer, loader_train, loader_val, epochs=1):
    model = model.to(device=device)

    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print("Iteration %d, loss = %.4f" % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()