import torch.nn as nn
import torch.nn.functional as F


class ELU(nn.Module):
    r"""
    Implementation of Exponential Linear Unit. https://arxiv.org/abs/1511.07289

    :param float alpha: the α value for the ELU. Default: 1.0
    :param bool inplace: In-place operation. Default: False

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.ELU()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([ 1.0000, -0.8647,  0.0000,  3.0000])
    """

    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super(ELU, self).__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        x = F.elu(x, alpha=self.alpha, inplace=self.inplace)
        return x
