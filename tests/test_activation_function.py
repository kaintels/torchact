import sys
import torch
import torch.nn as nn
from torchact import *
from torchact import __all__
import pytest


@pytest.mark.skip
def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def test_tensor_type():
    x = torch.rand(1, 5)

    for activation_name in __all__:
        test_model = nn.Sequential(nn.Linear(5, 3))
        test_model.add_module(activation_name, str_to_class(activation_name)())
        out_x = test_model(x)
        assert type(out_x) == torch.Tensor
        del test_model


def test_has_attr():
    for activation_name in __all__:
        if activation_name == "Softmax" or activation_name == "LogSoftmax":
            assert hasattr(str_to_class(activation_name)(), "dim")
        else:
            pass
