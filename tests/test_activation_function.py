import torch
import torch.nn as nn
from torchact import ReLU, SinLU, Softmax
import pytest

test_model = nn.Sequential(nn.Linear(5, 3))
test_model.add_module("ReLU", ReLU())
test_model.add_module("SinLU", SinLU())
test_model.add_module("Softmax", Softmax())
test_model.add_module("Output", nn.Linear(3, 1))


def test_tensor_type():
    x = torch.rand(1, 5)
    out_x = test_model(x)
    assert type(out_x) == torch.Tensor


def test_has_attr():
    assert Softmax().__getstate__()
    assert hasattr(Softmax(), "dim")
