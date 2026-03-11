"""
pycheron.plot.shap_plots — SHAP summary and waterfall plots.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pycheron.train.trained_model import TrainedModel


def shap_summary(
    model: "TrainedModel",
    X,
    n_samples: int = 200,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    SHAP summary (beeswarm) plot — shows global feature importance with direction.

    Parameters
    ----------
    model     : trained pycheron model
    X         : feature data (array or DataFrame)
    n_samples : max rows to use for SHAP computation
    figsize   : figure size
    save_path : save figure here
    show      : call plt.show()

    Examples
    --------
    >>> pycrn.plot.shap_summary(model, X_test)
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("shap and matplotlib required: pip install shap matplotlib")

    import numpy as np
    if hasattr(X, "values"):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    if len(X_arr) > n_samples:
        idx = np.random.choice(len(X_arr), n_samples, replace=False)
        X_arr = X_arr[idx]

    estimator = model.estimator
    feature_names = model.feature_names or [f"f{i}" for i in range(X_arr.shape[1])]

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
    elif hasattr(estimator, "coef_"):
        explainer = shap.LinearExplainer(estimator, X_arr)
    else:
        bg = shap.sample(X_arr, min(50, len(X_arr)))
        explainer = shap.KernelExplainer(estimator.predict, bg)

    shap_values = explainer.shap_values(X_arr)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class for binary

    plt.figure(figsize=figsize)
    shap.summary_plot(
        shap_values, X_arr,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"SHAP Summary — {model.algorithm}", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def shap_waterfall(
    model: "TrainedModel",
    X,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    SHAP waterfall plot for a single prediction — explains one row.

    Parameters
    ----------
    model      : trained pycheron model
    X          : feature data
    sample_idx : which row to explain (default 0)
    save_path  : save figure here
    show       : call plt.show()

    Examples
    --------
    >>> pycrn.plot.shap_waterfall(model, X_test, sample_idx=5)
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("shap and matplotlib required.")

    import numpy as np
    if hasattr(X, "values"):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    estimator = model.estimator
    feature_names = model.feature_names or [f"f{i}" for i in range(X_arr.shape[1])]

    if hasattr(estimator, "feature_importances_"):
        explainer = shap.TreeExplainer(estimator)
        sv = explainer(X_arr)
    else:
        bg = shap.sample(X_arr, min(50, len(X_arr)))
        explainer = shap.KernelExplainer(estimator.predict, bg)
        sv = explainer(X_arr[sample_idx:sample_idx+1])

    shap.waterfall_plot(sv[sample_idx], show=False)
    plt.title(f"SHAP Waterfall — {model.algorithm} (sample {sample_idx})", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
