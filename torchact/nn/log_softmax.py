import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LogSoftmax(nn.Module):
    r"""
    Implementation of Log(Softmax(x)).

    :param int dim: LogSoftmax dimension. Default: None

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.LogSoftmax()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([-2.1755, -5.1755, -3.1755, -0.1755])
    """

    def __init__(self, dim: Optional[int] = None):
        super(LogSoftmax, self).__init__()
        if not hasattr(self, "dim"):
            self.dim = None
        self.dim = dim

    def forward(self, x):
        x = torch.log(F.softmax(x, dim=self.dim))
        return x
