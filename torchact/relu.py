from .config import *
import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.Module):
    def __init__(self, inplace=FALSE_CONDITION):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        x = F.relu(x, inplace=self.inplace)
        return x
