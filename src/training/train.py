import torch
import torch.nn.functional as F

from config import device, dtype, print_every
from training.evaluator import evaluate


def train(model, optimizer, loader_train, loader_val,
          epochs=1, scheduler=None, early_stop_patience=0,
          start_epoch=1, best_acc=0.0,
          logger=None, recorder=None):
    """
    训练 model，返回 {"best_acc": float, "last_epoch": int}。
    """
    model = model.to(device=device)
    best_state = None
    epochs_no_improve = 0
    last_epoch = start_epoch - 1

    def log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    for e in range(start_epoch, epochs + 1):
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

        lr_now = optimizer.param_groups[0]["lr"]
        if recorder is not None:
            recorder.log(e, avg_loss, val_acc, lr_now)

        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            log("  LR -> %.6g" % optimizer.param_groups[0]["lr"])

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                log("Early stopping at epoch %d (no improve for %d epochs)"
                    % (e, epochs_no_improve))
                last_epoch = e
                break

        last_epoch = e

    if best_state is not None:
        model.load_state_dict(best_state)

    log("Best val accuracy = %.4f" % best_acc)
    return {"best_acc": best_acc, "last_epoch": last_epoch}