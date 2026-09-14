import argparse
from datetime import datetime

import torch
import torch.optim as optim

from config import TrainConfig
from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.deep_convnet import DeepConvNet
from training.train import train
from training.evaluator import evaluate
from utils.path import CHECKPOINTS_PATH, OUTPUTS_PATH
from utils.logger import get_logger, CSVRecorder

from data.datasets import build_loaders


EXPERIMENTS = {
    "mlp": (MLP, 1e-2),
    "shallow_convnet": (ShallowConvNet, 1e-2),
    "deep_convnet": (DeepConvNet, 0.1),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument("--experiment", type=str, default="mlp",
                        choices=list(EXPERIMENTS.keys()))
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--no-nesterov", action="store_true")
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


def run_experiment(cfg, dataset_name, batch_size):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "%s_%s" % (cfg.experiment, timestamp)

    log_path = OUTPUTS_PATH / ("%s.log" % base)
    csv_path = OUTPUTS_PATH / ("%s.csv" % base)
    ckpt_path = CHECKPOINTS_PATH / ("%s.pt" % base)

    logger = get_logger(cfg.experiment, log_path)
    recorder = CSVRecorder(csv_path)

    loader_train, loader_val, loader_test = build_loaders(
        name=dataset_name, batch_size=batch_size,
    )

    model_cls, _ = EXPERIMENTS[cfg.experiment]
    model = model_cls()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        nesterov=cfg.nesterov,
    )

    logger.info("=" * 60)
    logger.info("Experiment: %s", cfg.experiment)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Config: %s", cfg)
    logger.info("=" * 60)

    train(model, optimizer, loader_train, loader_val,
          epochs=cfg.epochs, logger=logger, recorder=recorder)

    test_acc = evaluate(model, loader_test)
    logger.info("Test accuracy = %.4f" % test_acc)

    torch.save(model.state_dict(), ckpt_path)
    logger.info("Saved to %s", ckpt_path)


def main():
    args = parse_args()
    cfg = build_config(args)
    run_experiment(cfg, dataset_name=args.dataset, batch_size=args.batch_size)


if __name__ == "__main__":
    main()