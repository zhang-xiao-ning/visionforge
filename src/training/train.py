import torch
import torch.nn.functional as F

from config import device, dtype, print_every
from training.evaluator import evaluate


def train(model, optimizer, loader_train, loader_val, epochs=1,
    logger=None, recorder=None):
    model = model.to(device=device)
    best_acc = 0.0
    best_state = None

    def log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    for e in range(1, epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

            if t % print_every == 0:
                log("Epoch %d, Iter %d, loss = %.4f" % (e, t, loss.item()))

        avg_loss = running_loss / n_batches
        val_acc = evaluate(model, loader_val)

        log("Epoch %d done. avg_train_loss = %.4f, val_acc = %.4f"
            % (e, avg_loss, val_acc))

        if recorder is not None:
            recorder.log(e, avg_loss, val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    log("Best val accuracy = %.4f" % best_acc)
    return best_acc