import torch.nn as nn
import torch.nn.functional as F


class ReLU(nn.Module):
    r"""
    Implementation of Rectified Linear Unit.

    :math:`\text{ReLU}(x) = \max(0, x)`

    :param bool inplace: In-place operation. Default: False

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.ReLU()
        >>> input = torch.tensor([1.0, -2.0, 0.0, 3.0])
        >>> output = m(input)
        >>> print(output)
        tensor([1., 0., 0., 3.])
    """

    def __init__(self, inplace: bool = False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        x = F.relu(x, inplace=self.inplace)
        return x
