import torch.nn as nn
from models.functional import flatten

class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)