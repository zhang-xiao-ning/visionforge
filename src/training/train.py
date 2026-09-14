import torch
import torch.nn.functional as F

from config import device, dtype, print_every
from training.evaluator import evaluate


def train(model, optimizer, loader_train, loader_val, epochs=1):
    """
    训练 model 并在每个 epoch 结束时评估。
    返回最优验证准确率。
    """
    model = model.to(device=device)
    best_acc = 0.0
    best_state = None

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
                print("Epoch %d, Iter %d, loss = %.4f" % (e, t, loss.item()))

        # 每个 epoch 结束，评估一次
        val_acc = evaluate(model, loader_val)
        print("Epoch %d done. Val accuracy = %.4f" % (e, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            # 深拷贝一份当前权重
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 恢复最优权重
    if best_state is not None:
        model.load_state_dict(best_state)

    print("Best val accuracy = %.4f" % best_acc)
    return best_acc