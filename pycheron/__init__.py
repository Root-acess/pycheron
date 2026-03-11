"""
pycheron — A simple, powerful Python ML framework.

Train a model in one line:
    >>> model = pycheron.train("data.csv", target="label")

Or let AutoML do everything:
    >>> model = pycheron.auto_train("data.csv")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "pycheron contributors"

from pycheron._api import train, auto_train, load_model, explain
from pycheron.config.settings import config
from pycheron.utils.logging import get_logger

__all__ = [
    "train",
    "auto_train",
    "load_model",
    "explain",
    "config",
    "get_logger",
    "__version__",
]
