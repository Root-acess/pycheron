"""
pycheron.plot.feature_importance — Feature importance bar chart.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pycheron.train.trained_model import TrainedModel


def feature_importance(
    model: "TrainedModel",
    top_n: int = 20,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Bar chart of feature importances from a trained model.

    Parameters
    ----------
    model    : TrainedModel from pycrn.train() or pycrn.auto_train()
    top_n    : show top N features (default 20)
    title    : chart title (auto-generated if None)
    figsize  : figure size tuple
    save_path: save figure to this path if given
    show     : call plt.show() (default True)

    Examples
    --------
    >>> pycrn.plot.feature_importance(model)
    >>> pycrn.plot.feature_importance(model, top_n=10, save_path="fi.png")
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    fi = None
    if model.evaluation and model.evaluation.feature_importances:
        fi = model.evaluation.feature_importances

    if fi is None:
        # Try to extract from estimator directly
        estimator = model.estimator
        names = model.feature_names or [f"f{i}" for i in range(100)]
        if hasattr(estimator, "feature_importances_"):
            import numpy as np
            vals = estimator.feature_importances_
            fi = dict(zip(names, vals.tolist()))
        elif hasattr(estimator, "coef_"):
            import numpy as np
            coef = np.abs(estimator.coef_).flatten()
            fi = dict(zip(names[:len(coef)], coef.tolist()))

    if not fi:
        print("No feature importances available for this model.")
        return

    # Sort and take top N
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, scores = zip(*sorted_fi)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn(
        [s / max(scores) for s in scores]
    )
    bars = ax.barh(range(len(features)), scores, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(title or f"Feature Importance — {model.algorithm} (top {len(features)})")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + max(scores) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va="center", fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
