from .__config__ import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class ABReLU(nn.Module):
    r"""
    Implementation of Average-Biased Rectified Linear Unit. https://arxiv.org/abs/1804.02051

    :param float alpha: parameter to be set empirically. Default: 1.0
    :param bool inplace: In-place operation. Default: False

    Examples::
        >>> import torch, torchact
        >>> m = torchact.ABReLU()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([0.5000, 0.0000, 0.0000, 2.5000])
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = FALSE_CONDITION):
        super(ABReLU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        if len(x.shape) < 2:
            beta = self.alpha * torch.mean(x)
        else:
            beta = self.alpha * torch.mean(x, -2)
        x_out = x - beta
        return torch.clip(x_out, min=0)