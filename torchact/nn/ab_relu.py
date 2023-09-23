import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.util import _value_is_not_nan


class ABReLU(nn.Module):
    r"""
    Implementation of Average-Biased Rectified Linear Unit https://arxiv.org/abs/1804.02051

    :param float alpha: parameter to be set empirically. Default: 1.0
    :param bool inplace: In-place operation. Default: False

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.ABReLU()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([0.1667, 0.0000, 0.0000, 0.8333])
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super(ABReLU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        if x.shape[0] > 1:
            beta = self.alpha * torch.mean(x, 0)
        else:
            beta = self.alpha * torch.mean(x)
        x_out = x - beta
        x_out = torch.clip(x_out, min=0)
        if x.shape[0] > 1:
            res = x_out / torch.sum(x_out, 0)
        else:
            res = x_out / torch.sum(x_out)
        _value_is_not_nan(res, 3)
        return res
