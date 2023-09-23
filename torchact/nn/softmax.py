import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Softmax(nn.Module):
    r"""
    Implementation of Softmax.

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    :param int dim: Softmax dimension. Default: None

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.Softmax()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([0.1135, 0.0057, 0.0418, 0.8390])
    """

    def __init__(self, dim: Optional[int] = None):
        super(Softmax, self).__init__()
        if not hasattr(self, "dim"):
            self.dim = None
        self.dim = dim

    def forward(self, x):
        x = F.softmax(x, dim=self.dim)
        return x
