"""
pycheron.plot.confusion_matrix — Styled confusion matrix heatmap.
"""

from __future__ import annotations
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from pycheron.train.trained_model import TrainedModel


def confusion_matrix(
    model: "TrainedModel",
    X_test,
    y_test,
    class_names: Optional[List[str]] = None,
    figsize: tuple = (8, 6),
    cmap: str = "Blues",
    normalize: bool = False,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a styled confusion matrix.

    Parameters
    ----------
    model       : trained pycheron model
    X_test      : test features (array or DataFrame)
    y_test      : true labels
    class_names : list of class label names
    figsize     : figure dimensions
    cmap        : colormap (default 'Blues')
    normalize   : show row-normalized percentages
    save_path   : save to file path if given
    show        : call plt.show()

    Examples
    --------
    >>> pycrn.plot.confusion_matrix(model, X_test, y_test)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix as sk_cm
    except ImportError:
        raise ImportError("matplotlib and scikit-learn required.")

    y_pred = model.estimator.predict(X_test)
    cm = sk_cm(y_test, y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n_classes = cm.shape[0]
    if class_names is None:
        if hasattr(model.preprocessor, "_label_encoder") and \
           model.preprocessor._label_encoder is not None:
            class_names = list(model.preprocessor._label_encoder.classes_)
        else:
            class_names = [str(i) for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap(cmap))
    plt.colorbar(im, ax=ax)

    tick_marks = range(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else str(int(val))
            ax.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=11, fontweight="bold",
            )

    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    fmt = "(normalized)" if normalize else ""
    ax.set_title(f"Confusion Matrix {fmt} — {model.algorithm}", fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
