import argparse
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig, device
from data.datasets import build_loaders
from models.deep_convnet import DeepConvNet
from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.vit import ViT
from training.evaluator import evaluate
from training.train import train
from utils.env import format_env_info, get_env_info
from utils.logger import CSVRecorder, get_logger
from utils.path import CHECKPOINTS_PATH, OUTPUTS_PATH
from utils.seed import set_seed

EXPERIMENTS = {
    "mlp": (MLP, 1e-2),
    "shallow_convnet": (ShallowConvNet, 1e-2),
    "deep_convnet": (DeepConvNet, 0.1),
    "vit": (ViT, 3e-4),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument("--experiment", type=str, default="mlp", choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no-nesterov", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--lr-scheduler", type=str, default="none", choices=["none", "step", "cosine"]
    )
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--early-stop-patience", type=int, default=0)

    parser.add_argument(
        "--resume", type=str, default=None, help="path to checkpoint to resume from"
    )
    parser.add_argument("--amp", action="store_true", help="enable mixed precision (CUDA only)")

    return parser.parse_args()


def build_config(args):
    _, default_lr = EXPERIMENTS[args.experiment]
    lr = args.learning_rate if args.learning_rate is not None else default_lr
    return TrainConfig(
        experiment=args.experiment,
        epochs=args.epochs,
        learning_rate=lr,
        momentum=args.momentum,
        nesterov=not args.no_nesterov,
        seed=args.seed,
        lr_scheduler=args.lr_scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        early_stop_patience=args.early_stop_patience,
        amp=args.amp,
    )


def build_scheduler(optimizer, cfg):
    if cfg.lr_scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )
    if cfg.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
        )
    return None


def run_experiment(cfg, dataset_name, batch_size, resume_path=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{cfg.experiment}_{timestamp}"

    log_path = OUTPUTS_PATH / (f"{base}.log")
    ckpt_path = CHECKPOINTS_PATH / (f"{base}.pt")
    cfg_path = OUTPUTS_PATH / (f"{base}.json")

    # CSV：resume 时复用旧的，否则新建
    if resume_path is not None:
        old_base = Path(resume_path).stem
        csv_path = OUTPUTS_PATH / (f"{old_base}.csv")
        csv_append = True
    else:
        csv_path = OUTPUTS_PATH / (f"{base}.csv")
        csv_append = False

    logger = get_logger(cfg.experiment, log_path)
    recorder = CSVRecorder(csv_path, append=csv_append)
    tb_dir = OUTPUTS_PATH / base
    writer = SummaryWriter(log_dir=str(tb_dir))

    snapshot = {
        "config": dataclasses.asdict(cfg),
        "dataset": dataset_name,
        "batch_size": batch_size,
        "env": get_env_info(),
        "resume_from": resume_path,
    }
    with open(cfg_path, "w") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    loader_train, loader_val, loader_test = build_loaders(
        name=dataset_name,
        batch_size=batch_size,
    )

    model_cls, _ = EXPERIMENTS[cfg.experiment]
    model = model_cls()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )
    scheduler = build_scheduler(optimizer, cfg)

    start_epoch = 1
    best_acc = 0.0

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        logger.info("Resumed from %s", resume_path)
        logger.info("Resume at epoch %d, best_acc = %.4f", start_epoch, best_acc)

    logger.info("=" * 60)
    logger.info("Experiment: %s", cfg.experiment)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Config: %s", cfg)
    logger.info(format_env_info())
    logger.info("=" * 60)

    result = train(
        model,
        optimizer,
        loader_train,
        loader_val,
        epochs=cfg.epochs,
        scheduler=scheduler,
        early_stop_patience=cfg.early_stop_patience,
        start_epoch=start_epoch,
        best_acc=best_acc,
        logger=logger,
        recorder=recorder,
        use_amp=cfg.amp,
        writer=writer,
    )

    test_acc = evaluate(model, loader_test)
    logger.info(f"Test accuracy = {test_acc:.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": result["last_epoch"],
            "best_acc": result["best_acc"],
            "config": dataclasses.asdict(cfg),
        },
        ckpt_path,
    )
    logger.info("Saved to %s", ckpt_path)
    writer.close()


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)
    run_experiment(
        cfg, dataset_name=args.dataset, batch_size=args.batch_size, resume_path=args.resume
    )


if __name__ == "__main__":
    main()
