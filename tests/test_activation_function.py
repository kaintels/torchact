import torch
import pytest
import torch.nn as nn
from torchact import ReLU, SinLU

model = nn.Sequential(
    nn.Linear(5, 3),
    ReLU(),
    SinLU(),
    nn.Linear(3, 1)
)


def test_tensor_type():
    x = torch.rand(1, 5)
    out_x = model(x)
    assert type(out_x) == torch.Tensor
