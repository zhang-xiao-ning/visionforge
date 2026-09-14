import torch.optim as optim

from models.mlp import MLP
from models.shallow_convnet import ShallowConvNet
from models.deep_convnet import DeepConvNet
from training.train import train_part34

import data.cifar10 as cifar10


def build_loaders():
    loader_train = cifar10.build_train_loader()
    loader_val = cifar10.build_val_loader()
    loader_test = cifar10.build_test_loader()
    return loader_train, loader_val, loader_test


def experiment_mlp(loader_train, loader_val):
    model = MLP()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    train_part34(model, optimizer, loader_train, loader_val)


def experiment_shallow_convnet(loader_train, loader_val):
    model = ShallowConvNet()
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    train_part34(model, optimizer, loader_train, loader_val)


def experiment_deep_convnet(loader_train, loader_val):
    model = DeepConvNet()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    train_part34(model, optimizer, loader_train, loader_val, epochs=10)


def main():
    loader_train, loader_val, loader_test = build_loaders()
    # experiment_mlp(loader_train, loader_val)
    # experiment_shallow_convnet(loader_train, loader_val)
    experiment_deep_convnet(loader_train, loader_val)


if __name__ == "__main__":
    main()