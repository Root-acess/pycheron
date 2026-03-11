"""
pycheron — The all-in-one Python ML framework.

Quick start
-----------
>>> import pycheron as pycrn

# pandas and numpy via pycheron
>>> df = pycrn.pd.read_smart("data.csv")
>>> arr = pycrn.np.normalize(df["col"].values)

# Train a model in one line
>>> model = pycrn.train("data.csv", target="label")
>>> model.evaluate()
>>> model.predict("test.csv")

# Full AutoML
>>> model = pycrn.auto_train("data.csv", time_budget=60)
>>> pycrn.plot.feature_importance(model)

# Experiment tracking
>>> pycrn.tracking.log("run_1", model)
>>> pycrn.tracking.leaderboard()

# Serve as REST API
>>> pycrn.infer.serve(model, port=8000)
"""

from __future__ import annotations

from pycheron._version import __version__

__author__ = "pycheron contributors"

# ── Wrapped third-party libraries (pycrn.pd, pycrn.np, etc.) ─────────────────
from pycheron.libs import _pandas as pd        # pycrn.pd.read_csv / pycrn.pd.read_smart
from pycheron.libs import _numpy as np         # pycrn.np.array   / pycrn.np.normalize
from pycheron.libs import _sklearn as sklearn  # pycrn.sklearn.metrics / .model_selection

# ── Core training API ─────────────────────────────────────────────────────────
from pycheron._api import train, auto_train, load_model, explain

# ── Submodule namespaces ──────────────────────────────────────────────────────
from pycheron import data        # pycrn.data.load / .profile / .clean / .split
from pycheron import preprocess  # pycrn.preprocess.encode / .scale / .select_features
from pycheron import evaluate    # pycrn.evaluate.compare
from pycheron import plot        # pycrn.plot.feature_importance / .confusion_matrix
from pycheron import tracking    # pycrn.tracking.log / .leaderboard
from pycheron import infer       # pycrn.infer.serve

# ── Config & logging ──────────────────────────────────────────────────────────
from pycheron.config.settings import config
from pycheron.utils.logging import get_logger

__all__ = [
    # version
    "__version__",
    # wrapped libs
    "pd",
    "np",
    "sklearn",
    # core API
    "train",
    "auto_train",
    "load_model",
    "explain",
    # submodules
    "data",
    "preprocess",
    "evaluate",
    "plot",
    "tracking",
    "infer",
    # config
    "config",
    "get_logger",
]
