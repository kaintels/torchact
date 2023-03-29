# torchact

<div align="center">

TorchAct, collection of activation function for PyTorch.

---

| ![image](https://img.shields.io/badge/-Tests:-black?style=flat-square) [![CI](https://github.com/kaintels/torchact/actions/workflows/ci.yml/badge.svg)](https://github.com/kaintels/torchact/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/kaintels/torchact/branch/main/graph/badge.svg?token=EJMC8R0OOT)](https://codecov.io/gh/kaintels/torchact) |
|:---
| ![image](https://img.shields.io/badge/-Stable%20Releases:-black?style=flat-square) ![PyPI - Status](https://img.shields.io/pypi/status/torchact) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchact) [![image](https://badge.fury.io/py/torchact.svg)](https://badge.fury.io/py/torchact) [![Downloads](https://static.pepy.tech/badge/torchact)](https://pepy.tech/project/torchact)
| ![image](https://img.shields.io/badge/-Features:-black?style=flat-square) ![PyPI - License](https://img.shields.io/pypi/l/torchact?color=blue) [![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Quick Start

```python
import torch
import torch.nn as nn
from torchact import ReLU

model = nn.Sequential(
    nn.Linear(5, 3),
    ReLU(),
    nn.Linear(3, 1)
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

0. Install library using `requirements.txt`
1. Write your code in torchact folder.
2. Add your module in `__init__.py` (`__version__` cannot be changed. It will be decided later.)

For example.
```
from .your_module import Your_Module
__all__ = ("ReLU", "SinLU", "Softmax", "Your_Module")
```
3. Add your module in `test_activation_function.py`

For example.
```
from torchact import Your_Module
test_model.add_module("Your_Module", Your_Module())
```
4. Send a PR. Code testing happens automatically. (PYPI is upgraded by the admin himself.)