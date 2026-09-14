import torch.nn as nn
import torch.nn.functional as F

from models.functional import flatten

class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, 5, padding='same')
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding='same')
        nn.init.kaiming_normal_(self.conv2.weight)
        self.fc = nn.Linear(channel_2*32*32, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        scores = None
        a1 = F.relu(self.conv1(x))
        a2 = F.relu(self.conv2(a1))
        scores = self.fc(flatten(a2))
        return scores