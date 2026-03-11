"""
pycheron.plot.learning_curve — Train vs validation score over training size.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pycheron.train.trained_model import TrainedModel


def learning_curve(
    model: "TrainedModel",
    X,
    y,
    cv: int = 5,
    scoring: Optional[str] = None,
    train_sizes=None,
    figsize: tuple = (9, 5),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a learning curve: train and CV score vs training set size.

    Parameters
    ----------
    model      : trained pycheron model
    X          : feature matrix
    y          : labels
    cv         : cross-validation folds
    scoring    : sklearn scoring string (auto-selected if None)
    train_sizes: array-like of fractions (default linspace(0.1, 1.0, 8))
    figsize    : figure dimensions
    save_path  : save to file
    show       : call plt.show()

    Examples
    --------
    >>> pycrn.plot.learning_curve(model, X_train, y_train)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.model_selection import learning_curve as sk_lc
    except ImportError:
        raise ImportError("matplotlib and scikit-learn required.")

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)

    if scoring is None:
        scoring = "roc_auc" if model.task.is_classification else "r2"

    train_sz, train_scores, val_scores = sk_lc(
        model.estimator, X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_sz, train_mean, "o-", color="#2196F3", label="Train score")
    ax.fill_between(train_sz, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#2196F3")
    ax.plot(train_sz, val_mean, "s-", color="#4CAF50", label=f"CV score ({cv}-fold)")
    ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#4CAF50")

    ax.set_xlabel("Training set size", fontsize=11)
    ax.set_ylabel(scoring, fontsize=11)
    ax.set_title(f"Learning Curve — {model.algorithm}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
