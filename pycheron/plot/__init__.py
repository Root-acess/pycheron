"""
pycheron.plot — One-line ML visualizations.

Usage:
    import pycheron as pycrn

    pycrn.plot.feature_importance(model)
    pycrn.plot.confusion_matrix(model, X_test, y_test)
    pycrn.plot.learning_curve(model, X, y)
    pycrn.plot.distribution(df, column="age")
    pycrn.plot.correlation(df)
    pycrn.plot.shap_summary(model, X)
"""

from __future__ import annotations

from pycheron.plot.feature_importance import feature_importance
from pycheron.plot.confusion_matrix import confusion_matrix
from pycheron.plot.learning_curve import learning_curve
from pycheron.plot.data_plots import distribution, correlation, class_balance
from pycheron.plot.shap_plots import shap_summary, shap_waterfall

__all__ = [
    "feature_importance",
    "confusion_matrix",
    "learning_curve",
    "distribution",
    "correlation",
    "class_balance",
    "shap_summary",
    "shap_waterfall",
]
