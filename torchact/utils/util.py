import warnings
import torch


def _value_is_not_nan(x: torch.Tensor, stacklevel: int) -> bool:
    if torch.isnan(x).any().item() == True:
        warnings.warn(
            "The tensor value have nan. check your code. ",
            stacklevel=stacklevel,
        )
        ret = False
    else:
        ret = True
    return ret
