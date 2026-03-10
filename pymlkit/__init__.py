"""
pymlkit — A simple, powerful Python ML framework.

Train a model in one line:
    >>> model = pymlkit.train("data.csv", target="label")

Or let AutoML do everything:
    >>> model = pymlkit.auto_train("data.csv")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "pymlkit contributors"

from pymlkit._api import train, auto_train, load_model, explain
from pymlkit.config.settings import config
from pymlkit.utils.logging import get_logger

__all__ = [
    "train",
    "auto_train",
    "load_model",
    "explain",
    "config",
    "get_logger",
    "__version__",
]
