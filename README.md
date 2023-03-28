# torchact

<div align="center">

TorchAct, collection of activation function for PyTorch.

---
![PyPI - Status](https://img.shields.io/pypi/status/torchact)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torchact)
[![image](https://badge.fury.io/py/torchact.svg)](https://badge.fury.io/py/torchact)
![PyPI - License](https://img.shields.io/pypi/l/torchact?color=blue)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
> pip install torchact
```