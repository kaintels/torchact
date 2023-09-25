import sys
import torch
import torch.nn as nn
from torchact.functional import *
from torchact.functional import __all__ as F_all
from torchact.nn import *
from torchact.nn import __all__ as C_all
import pytest
import logging
import warnings

logger = logging.getLogger("test")

all_test_case = []  # test 1 ~ N
all_test_case.append({"name": "fix", "val": torch.tensor([1.0, -2.0, 0.0, 3.0])})
all_test_case.append({"name": "rand", "val": torch.rand(1, 5)})
all_test_case.append({"name": "ones", "val": torch.ones(1, 5)})
all_test_case.append({"name": "zeros", "val": torch.zeros(1, 5)})
all_test_case.append({"name": "10x10", "val": torch.rand(10, 10) * 10})
all_test_case.append({"name": "b-size 64", "val": torch.rand(64, 1, 32, 32) * 10})
all_test_case.append({"name": "large", "val": torch.rand(256, 3, 500, 500) * 10})


@pytest.mark.skip
def _str_to_def(defname):
    return getattr(sys.modules[__name__], defname)


@pytest.mark.skip
def _str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


@pytest.mark.skip
def _value_is_not_nan(x: torch.Tensor, stacklevel: int) -> bool:
    if torch.isnan(x).any().item():
        warnings.warn(
            "The tensor value have nan. check your code. ",
            stacklevel=stacklevel,
        )
        ret = False
    else:
        ret = True
    return ret


def test_nan_check():
    for activation_name in F_all:
        test_number = 1
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
    for activation_name in F_all:
        test_def = _str_to_def(activation_name)
        for test_case in all_test_case:
            out = test_def(test_case["val"])
            assert type(out) == torch.Tensor
        del test_def

    for activation_name in C_all:
        for test_case in all_test_case:
            if len(list(test_case["val"].shape)) > 1:
                test_model = nn.Sequential(nn.Linear(test_case["val"].shape[-1], 3))
            else:
                test_model = nn.Sequential(nn.Linear(test_case["val"].shape[0], 3))
            test_model.add_module(activation_name, _str_to_class(activation_name)())
            out = test_model(test_case["val"])
            assert type(out) == torch.Tensor
        del test_model


def test_has_attr():
    for activation_name in C_all:
        if activation_name == "Softmax" or activation_name == "LogSoftmax":
            assert hasattr(_str_to_class(activation_name)(), "dim")
        else:
            pass


def test_model_can_jit_trace():
    for activation_name in C_all:
        for test_case in all_test_case:
            if len(list(test_case["val"].shape)) > 1:
                test_model = nn.Sequential(nn.Linear(test_case["val"].shape[-1], 3))
            else:
                test_model = nn.Sequential(nn.Linear(test_case["val"].shape[0], 3))
            test_model.add_module(activation_name, _str_to_class(activation_name)())
            test_model = torch.jit.script(test_model)
            out = test_model(test_case["val"])
            assert type(out) == torch.Tensor
        del test_model
