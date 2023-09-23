# torchact

<div align="center">

TorchAct, collection of activation function for PyTorch.

---

| ![image](https://img.shields.io/badge/-Tests:-black?style=flat-square) [![CI](https://github.com/kaintels/torchact/actions/workflows/ci.yml/badge.svg)](https://github.com/kaintels/torchact/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/kaintels/torchact/branch/main/graph/badge.svg?token=EJMC8R0OOT)](https://codecov.io/gh/kaintels/torchact) [![Read the Docs](https://img.shields.io/readthedocs/torchact)](https://torchact.readthedocs.io/) |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
| ![image](https://img.shields.io/badge/-Stable%20Releases:-black?style=flat-square) ![PyPI - Status](https://img.shields.io/pypi/status/torchact) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchact) [![image](https://badge.fury.io/py/torchact.svg)](https://badge.fury.io/py/torchact) [![Downloads](https://static.pepy.tech/badge/torchact)](https://pepy.tech/project/torchact)  
| ![image](https://img.shields.io/badge/-Features:-black?style=flat-square) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?&logo=PyTorch&logoColor=white) ![PyPI - License](https://img.shields.io/pypi/l/torchact?color=blue) [![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Quick Start

```python
import torch
import torch.nn as nn
import torchact.nn as actnn

model = nn.Sequential(
    nn.Linear(5, 3),
    actnn.ReLU(),
    nn.Linear(3, 1),
    nn.Sigmoid()
)

dummy = torch.rand(1, 5)
print(model(dummy))
```

## Installation

```shell
pip install torchact
```

## How to Contribute

Thanks for your contribution!

There are several steps for contributing.

0. Fork this repo (you can work dev branch.)
1. Install library using `requirements.txt`
2. Write your code in torchact folder.
3. Add your module in `__init__.py` (`__version__` cannot be changed. It will be decided later.)

For example.

```python
from .your_module import Your_Module
__all__ = ("ReLU", "SinLU", "Softmax", "Your_Module")
```

3. If you want to test case, Write test case.

For example.

```python
def test_has_attr():
    for activation_name in __all__:
        if activation_name == "Softmax":
            assert hasattr(str_to_class(activation_name)(), "dim")
        else:
            pass
```

4. Run black style.`black .`
5. Send a PR. Code testing happens automatically. (PYPI is upgraded by the admin himself.)

## Citing TorchAct

To cite this repository:

```
@article{hantorchact,
  title={TorchAct, collection of activation function for PyTorch.},
  author={Seungwoo Han},
  publisher={Engineering Archive},
  doi={10.31224/2988},
  url={https://engrxiv.org/preprint/view/2988}
  year={2023}
}
```
