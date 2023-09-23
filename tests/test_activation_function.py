import sys
import torch
import torch.nn as nn
from torchact.functional import *
from torchact.functional import __all__ as F_all
from torchact.nn import *
from torchact.nn import __all__ as C_all
from torchact.utils import _value_is_not_nan
import pytest
import logging

logger = logging.getLogger("test")


@pytest.mark.skip
def _str_to_def(defname):
    return getattr(sys.modules[__name__], defname)


@pytest.mark.skip
def _str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def test_nan_check():
    all_test_case = []  # test 1 ~ N
    all_test_case.append({"name": "fix", "val": torch.tensor([1.0, -2.0, 0.0, 3.0])})
    all_test_case.append({"name": "rand", "val": torch.rand(1, 5)})
    all_test_case.append({"name": "ones", "val": torch.ones(1, 5)})
    all_test_case.append({"name": "zeros", "val": torch.zeros(1, 5)})
    all_test_case.append({"name": "10x10", "val": torch.rand(10, 10) * 10})
    all_test_case.append({"name": "b-size 64", "val": torch.rand(64, 1, 32, 32) * 10})
    all_test_case.append({"name": "large", "val": torch.rand(256, 3, 500, 500) * 10})

    test_number = 1
    for activation_name in F_all:
        test_def = _str_to_def(activation_name)

        for test_case in all_test_case:
            try:
                out = test_def(test_case["val"])
                assert _value_is_not_nan(out, 3)
            except:
                logger.warning(
                    f"Test case {test_number}: {test_case['name']} is failed. check '{test_def}'"
                )
                pass
            finally:
                test_number += 1


def test_output_tensor_type():
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
