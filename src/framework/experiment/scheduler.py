import math

import torch.optim as optim
from torch.optim import lr_scheduler

from framework.experiment.spec import TrainConfig


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> lr_scheduler.LRScheduler:
    """Build a single step-based LambdaLR.

    Warmup and main schedule compose freely: warmup is a linear ramp
    over `warmup_steps` optimizer steps, then the main schedule takes
    over on the post-warmup step index. Any of `const` / `cosine` /
    `step` can follow any warmup length.

    The returned mode is always "step": `train()` calls scheduler.step()
    after every optimizer step. `step_size` for StepLR is expressed in
    epochs and converted internally.
    """
    opt_steps_per_epoch = max(1, steps_per_epoch // cfg.accum_steps)
    total_steps = cfg.epochs * opt_steps_per_epoch

    def lr_lambda(step: int) -> float:
        if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)

        s = step - cfg.warmup_steps
        if cfg.lr_scheduler == "const":
            return 1.0
        if cfg.lr_scheduler == "cosine":
            remaining = max(1, total_steps - cfg.warmup_steps)
            progress = min(1.0, s / remaining)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if cfg.lr_scheduler == "step":
            every = max(1, cfg.step_size * opt_steps_per_epoch)
            return cfg.gamma ** (s // every)
        raise ValueError(f"Unknown lr_scheduler: {cfg.lr_scheduler}")

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
