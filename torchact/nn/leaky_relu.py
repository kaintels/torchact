import torch.nn as nn
import torch.nn.functional as F


class LeakyReLU(nn.Module):
    r"""
    Implementation of LeakyReLU.

    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative_slope} * \min(0, x)`

    :param float neg_slope: Angle of the negative slope. Default: 1e-2
    :param bool inplace: In-place operation. Default: False

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.LeakyReLU()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([ 1.0000, -0.0200,  0.0000,  3.0000])
    """

    def __init__(self, neg_slope: float = 1e-2, inplace: bool = False):
        super(LeakyReLU, self).__init__()
        self.neg_slope = neg_slope
        self.inplace = inplace

    def forward(self, x):
        x = F.leaky_relu(x, negative_slope=self.neg_slope, inplace=self.inplace)
        return x
