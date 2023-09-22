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
        tensor([0.1667, 0.0000, 0.0000, 0.8333])
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = FALSE_CONDITION):
        super(ABReLU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        beta = self.alpha * torch.mean(x, 0)
        x_out = x - beta
        x_out = torch.clip(x_out, min=0)
        res = x_out / torch.sum(x_out, 0)
        return res
