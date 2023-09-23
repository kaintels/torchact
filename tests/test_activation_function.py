import sys
import torch
import torch.nn as nn
from torchact.functional import *
from torchact.functional import __all__ as F_all
from torchact.nn import *
from torchact.nn import __all__ as C_all
from torchact.utils import _value_is_not_nan
import pytest

@pytest.mark.skip
def _str_to_def(defname):
    return getattr(sys.modules[__name__], defname)

@pytest.mark.skip
def _str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def test_nan_check():
    test_x1 = torch.tensor([1.0, -2.0, 0.0, 3.0])
    test_x2 = torch.rand(1, 5)
    test_x3 = torch.ones(1, 5)
    test_x4 = torch.zeros(1, 5)

    for activation_name in F_all:
        test_def = _str_to_def(activation_name)
        out_x1 = test_def(test_x1)
        assert _value_is_not_nan(out_x1, 3)
        out_x2 = test_def(test_x2)
        assert _value_is_not_nan(out_x2, 3)
        out_x3 = test_def(test_x3)
        assert _value_is_not_nan(out_x3, 3)
        out_x4 = test_def(test_x4)
        assert _value_is_not_nan(out_x4, 3)


def test_tensor_type():
    x = torch.rand(1, 5)

    for activation_name in F_all:
        test_def = _str_to_def(activation_name)
        out_x = test_def(x)
        assert type(out_x) == torch.Tensor
        del test_def

    for activation_name in C_all:
        test_model = nn.Sequential(nn.Linear(5, 3))
        test_model.add_module(activation_name, _str_to_class(activation_name)())
        out_x = test_model(x)
        assert type(out_x) == torch.Tensor
        del test_model


def test_has_attr():
    for activation_name in C_all:
        if activation_name == "Softmax" or activation_name == "LogSoftmax":
            assert hasattr(_str_to_class(activation_name)(), "dim")
        else:
            pass


if __name__ == "__main__":
    x = torch.ones(1, 5)
    for activation_name in F_all:
        test_def = _str_to_def(activation_name)
        out_x = test_def(x)
        print(out_x)
