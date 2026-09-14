import argparse

import torch
import torch.optim as optim

from config import TrainConfig
from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.deep_convnet import DeepConvNet
from training.train import train
from training.evaluator import evaluate
from utils.path import CHECKPOINTS_PATH

import data.cifar10 as cifar10


# 实验名 -> (模型类, 默认学习率)
EXPERIMENTS = {
    "mlp": (MLP, 1e-2),
    "shallow_convnet": (ShallowConvNet, 1e-2),
    "deep_convnet": (DeepConvNet, 0.1),
}


def build_loaders():
    loader_train = cifar10.build_train_loader()
    loader_val = cifar10.build_val_loader()
    loader_test = cifar10.build_test_loader()
    return loader_train, loader_val, loader_test


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument(
        "--experiment", type=str, default="mlp",
        choices=list(EXPERIMENTS.keys()),
        help="which experiment to run",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="if not set, use the experiment default",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--no-nesterov", action="store_true",
        help="disable Nesterov momentum",
    )
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
    )


def run_experiment(cfg, loaders):
    model_cls, _ = EXPERIMENTS[cfg.experiment]
    model = model_cls()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )

    loader_train, loader_val, loader_test = loaders

    print("=" * 60)
    print("Experiment:", cfg.experiment)
    print("Config:", cfg)
    print("=" * 60)

    train(model, optimizer, loader_train, loader_val, epochs=cfg.epochs)

    test_acc = evaluate(model, loader_test)
    print("Test accuracy = %.4f" % test_acc)

    save_path = CHECKPOINTS_PATH / ("%s.pt" % cfg.experiment)
    torch.save(model.state_dict(), save_path)
    print("Saved to", save_path)


def main():
    args = parse_args()
    cfg = build_config(args)
    loaders = build_loaders()
    run_experiment(cfg, loaders)


if __name__ == "__main__":
    main()