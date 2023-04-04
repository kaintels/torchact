import setuptools
import os
import re


def _read(f):
    with open(f, "r") as file:
        descript = file.read()
    return descript


def _read_version():
    regexp = re.compile(r"^__version__\W*=\W*([\d.abrc]+)")
    version = os.path.join(os.path.dirname(__file__), "torchact", "__init__.py")

    with open(version) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
    raise RuntimeError("Cannot find version in torchact/__init__.py")

    return version


setuptools.setup(
    name="torchact",
    version=_read_version(),
    author="Seungwoo Han",
    author_email="seungwoohan0108@gmail.com",
    description="TorchAct, collection of activation function for PyTorch.",
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/kaintels/torchact",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
)
