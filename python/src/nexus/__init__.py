"""Nexus Python kabugu.

Bu yazilim ekosistemi Bursa Teknik Universitesi Bilgisayar Muhendisligi
1. sinif ogrencisi Muhammed Fatih Sahin tarafindan yapay zeka destekli
olarak gelistirilmistir.
"""

import os


def _cuda_bin():
    cuda = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4")
    candidate = os.path.join(cuda, "bin", "x64")
    if os.path.isdir(candidate):
        os.add_dll_directory(candidate)


_cuda_bin()

from nexus._core import Tensor, __version__, load, save
from nexus._core import custom, data, linalg, loss, metrics, nn, optim, rl, train
from nexus.fnstore import remember, remember_module

_raw_activation = nn.activation
_raw_module = nn.PythonModule


def _activation(name, forward, backward=None):
    if name != "Swish":
        remember(name, forward, backward)
    if backward is None:
        return _raw_activation(name, forward)
    return _raw_activation(name, forward, backward)


nn.activation = _activation


def _python_module(name, forward, backward=None, shapes=None):
    remember_module(name, forward, backward, shapes or [])
    if backward is None:
        return _raw_module(name, forward, shapes=shapes or [])
    return _raw_module(name, forward, backward, shapes or [])


nn.PythonModule = _python_module

__all__ = [
    "Tensor",
    "__version__",
    "custom",
    "data",
    "linalg",
    "load",
    "loss",
    "metrics",
    "nn",
    "optim",
    "rl",
    "save",
    "train",
]
