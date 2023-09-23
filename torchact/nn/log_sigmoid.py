import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LogSigmoid(nn.Module):
    r"""
    Implementation of Log(Sigmoid).

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.LogSigmoid()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([-0.3133, -2.1269, -0.6931, -0.0486])
    """

    def __init__(self):
        super(LogSigmoid, self).__init__()

    def forward(self, x):
        x = torch.log(1 / (1 + torch.exp(-x)))
        return x
