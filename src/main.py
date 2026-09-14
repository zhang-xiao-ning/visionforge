import torch
import torch.optim as optim

from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.deep_convnet import DeepConvNet
from training.train import train
from training.evaluator import evaluate
from utils.path import CHECKPOINTS_PATH

import data.cifar10 as cifar10


def build_loaders():
    loader_train = cifar10.build_train_loader()
    loader_val = cifar10.build_val_loader()
    loader_test = cifar10.build_test_loader()
    return loader_train, loader_val, loader_test


def run_experiment(name, model, optimizer, loaders, epochs=1):
    loader_train, loader_val, loader_test = loaders

    print("=" * 60)
    print("Experiment:", name)
    print("=" * 60)

    train(model, optimizer, loader_train, loader_val, epochs=epochs)

    test_acc = evaluate(model, loader_test)
    print("Test accuracy = %.4f" % test_acc)

    save_path = CHECKPOINTS_PATH / ("%s.pt" % name)
    torch.save(model.state_dict(), save_path)
    print("Saved to", save_path)
    print()


def main():
    loaders = build_loaders()

    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    run_experiment("mlp", model, optimizer, loaders)

    # model = ShallowConvNet()
    # optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    # run_experiment("shallow_convnet", model, optimizer, loaders)


if __name__ == "__main__":
    main()