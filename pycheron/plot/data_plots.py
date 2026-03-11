"""
pycheron.plot.data_plots — Distribution, correlation, and class balance plots.
"""

from __future__ import annotations
from typing import Optional, List
import pandas as pd


def distribution(
    df: pd.DataFrame,
    column: str,
    hue: Optional[str] = None,
    bins: int = 30,
    figsize: tuple = (9, 5),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Distribution plot for a single column (histogram + KDE).

    Parameters
    ----------
    df       : DataFrame
    column   : column to plot
    hue      : color by this column (e.g. target label)
    bins     : number of histogram bins
    figsize  : figure size
    save_path: save figure here
    show     : call plt.show()

    Examples
    --------
    >>> pycrn.plot.distribution(df, "age", hue="label")
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    fig, ax = plt.subplots(figsize=figsize)

    if hue and hue in df.columns:
        for label, group in df.groupby(hue):
            vals = group[column].dropna()
            ax.hist(vals, bins=bins, alpha=0.5, label=str(label), density=True)
        ax.legend(title=hue)
    else:
        vals = df[column].dropna()
        ax.hist(vals, bins=bins, alpha=0.7, color="#2196F3", density=True, label=column)
        # Add KDE
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            x = np.linspace(vals.min(), vals.max(), 300)
            ax.plot(x, kde(x), "r-", linewidth=2, label="KDE")
            ax.legend()
        except ImportError:
            pass

    ax.set_xlabel(column, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Distribution: {column}", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def correlation(
    df: pd.DataFrame,
    method: str = "pearson",
    figsize: tuple = (10, 8),
    cmap: str = "coolwarm",
    annot: bool = True,
    mask_upper: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Correlation heatmap for all numeric columns.

    Parameters
    ----------
    df          : DataFrame
    method      : 'pearson' | 'spearman' | 'kendall'
    figsize     : figure size
    cmap        : colormap
    annot       : annotate cells with values
    mask_upper  : show only lower triangle
    save_path   : save figure here
    show        : call plt.show()

    Examples
    --------
    >>> pycrn.plot.correlation(df)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("No numeric columns found for correlation matrix.")
        return

    corr = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)

    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    import matplotlib.colors as mcolors
    im = ax.imshow(
        corr.values,
        cmap=cmap,
        vmin=-1, vmax=1,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, shrink=0.8)

    n = len(corr.columns)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    if annot:
        for i in range(n):
            for j in range(n):
                if mask_upper and j > i:
                    continue
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_title(f"Correlation Matrix ({method})", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def class_balance(
    df: pd.DataFrame,
    target: str,
    figsize: tuple = (7, 5),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Bar chart showing class distribution of the target column.

    Examples
    --------
    >>> pycrn.plot.class_balance(df, target="label")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    counts = df[target].value_counts()
    pcts = df[target].value_counts(normalize=True) * 100

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(counts.index.astype(str), counts.values, color="#42A5F5", edgecolor="white")

    for bar, (label, pct) in zip(bars, pcts.items()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel(target, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Class Balance: {target}", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
