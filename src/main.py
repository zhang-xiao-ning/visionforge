import torch.nn as nn
import torch.optim as optim

from models.flatten import Flatten
from training.train import train_part34

import data.cifar10 as cifar10


def build_loaders():
    loader_train = cifar10.build_train_loader()
    loader_val = cifar10.build_val_loader()
    loader_test = cifar10.build_test_loader()
    return loader_train, loader_val, loader_test


def experiment_sequential_mlp(loader_train, loader_val):
    model = nn.Sequential(
        Flatten(),
        nn.Linear(3 * 32 * 32, 4000),
        nn.ReLU(),
        nn.Linear(4000, 10),
    )
    optimizer = optim.SGD(model.parameters(), lr=1e-2,
                          momentum=0.9, nesterov=True)
    train_part34(model, optimizer, loader_train, loader_val)


def experiment_sequential_convnet(loader_train, loader_val):
    c1, c2 = 32, 16
    model = nn.Sequential(
        nn.Conv2d(3, c1, 5, padding="same"),
        nn.ReLU(),
        nn.Conv2d(c1, c2, 3, padding="same"),
        nn.ReLU(),
        Flatten(),
        nn.Linear(c2 * 32 * 32, 10),
    )
    optimizer = optim.SGD(model.parameters(), lr=1e-2,
                          momentum=0.9, nesterov=True)
    train_part34(model, optimizer, loader_train, loader_val)


def experiment_deep_convnet(loader_train, loader_val):
    c1, c2, c3, c4, c5 = 16, 32, 32, 64, 64
    model = nn.Sequential(
        nn.Conv2d(3, c1, 3, stride=1, padding=1),
        nn.BatchNorm2d(c1),
        nn.ReLU(),
        nn.Conv2d(c1, c2, 3, stride=1, padding=1),
        nn.BatchNorm2d(c2),
        nn.ReLU(),
        nn.Conv2d(c2, c3, 3, stride=2, padding=1),
        nn.BatchNorm2d(c3),
        nn.ReLU(),
        nn.Conv2d(c3, c4, 3, stride=1, padding=1),
        nn.BatchNorm2d(c4),
        nn.ReLU(),
        nn.Conv2d(c4, c5, 3, stride=2, padding=1),
        nn.BatchNorm2d(c5),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(c5, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 10),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, nesterov=True)
    train_part34(model, optimizer, loader_train, loader_val, epochs=10)


def main():
    loader_train, loader_val, loader_test = build_loaders()
    # experiment_sequential_mlp(loader_train, loader_val)
    # experiment_sequential_convnet(loader_train, loader_val)
    experiment_deep_convnet(loader_train, loader_val)


if __name__ == "__main__":
    main()