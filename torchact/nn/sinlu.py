import torch
import torch.nn as nn


class SinLU(nn.Module):
    r"""
    Implementation of Sinu-Sigmoidal Linear Unit https://doi.org/10.3390/math10030337

    :math:`\text{SinLU}(x) = (x+a\text{sin}bx) \cdot \sigma(x)`

    :param float a: Amplitude of the sine. Default: 1.0
    :param float b: Frequency of the sine. Default: 1.0

    Examples::
        >>> import torch, torchact
        >>> m = torchact.nn.SinLU()
        >>> input = torch.randn(1, 3)
        >>> output = m(input)
        >>> print(output)
        tensor([[0.1167, 2.2457, 0.0896]], grad_fn=<MulBackward0>)
    """

    def __init__(self, a: float = 1.0, b: float = 1.0):
        super(SinLU, self).__init__()
        self.a = nn.Parameter(torch.FloatTensor([a]))
        self.b = nn.Parameter(torch.FloatTensor([b]))

    def forward(self, x):
        return (x + self.a * torch.sin(self.b * x)) * torch.sigmoid(x)
