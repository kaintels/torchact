import torch
import torch.nn as nn
import torch.nn.functional as F


class Step(nn.Module):
    r"""
    Implementation of binary step activation.

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.Step()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([1, 0, 0, 1])
    """

    def __init__(self):
        super(Step, self).__init__()

    def forward(self, x):
        x = torch.where(x > 0, 1, 0)
        return x
