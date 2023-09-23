import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Sigmoid(nn.Module):
    r"""
    Implementation of Sigmoid.

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.Sigmoid()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([0.7311, 0.1192, 0.5000, 0.9526])
    """

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        x = 1 / (1 + torch.exp(-x))
        return x
