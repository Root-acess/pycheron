"""
pycheron.libs._numpy — numpy + pycheron-specific helpers.

Usage:
    import pycheron as pycrn
    arr = pycrn.np.array([1, 2, 3])          # standard numpy
    arr = pycrn.np.normalize(arr)             # pycheron helper
    df  = pycrn.np.to_frame(arr)             # convert to DataFrame
"""

from __future__ import annotations

# Re-export everything from numpy so pycrn.np IS numpy
from numpy import *                             # noqa: F401, F403
import numpy as _np

from numpy import (                             # noqa: F401
    array, zeros, ones, empty, arange, linspace,
    mean, std, median, var, sum, min, max,
    dot, matmul, transpose, reshape, concatenate, stack, vstack, hstack,
    random, linalg, fft,
    float32, float64, int32, int64, bool_,
    inf, nan, pi, e,
)

from typing import Optional as _Optional, Union as _Union, Tuple as _Tuple


# ── pycheron-specific helpers ─────────────────────────────────────────────────

def normalize(
    arr: "ndarray",
    method: str = "minmax",
    axis: _Optional[int] = None,
) -> "ndarray":
    """
    Normalize an array.

    Parameters
    ----------
    arr    : input array
    method : 'minmax' (0-1 range) | 'zscore' (zero mean, unit variance)
             | 'l2' (unit norm per row)
    axis   : axis along which to normalize (None = whole array)

    Examples
    --------
    >>> arr = pycrn.np.normalize(arr, method="zscore")
    """
    if method == "minmax":
        mn = arr.min(axis=axis, keepdims=True)
        mx = arr.max(axis=axis, keepdims=True)
        denom = mx - mn
        denom = _np.where(denom == 0, 1, denom)
        return (arr - mn) / denom
    elif method == "zscore":
        mu = arr.mean(axis=axis, keepdims=True)
        sigma = arr.std(axis=axis, keepdims=True)
        sigma = _np.where(sigma == 0, 1, sigma)
        return (arr - mu) / sigma
    elif method == "l2":
        norm = _np.linalg.norm(arr, axis=axis, keepdims=True)
        norm = _np.where(norm == 0, 1, norm)
        return arr / norm
    else:
        raise ValueError(f"Unknown method '{method}'. Use: 'minmax', 'zscore', 'l2'")


def to_frame(
    arr: "ndarray",
    columns: _Optional[list] = None,
    index=None,
) -> "DataFrame":
    """
    Convert a numpy array to a pandas DataFrame.

    Examples
    --------
    >>> df = pycrn.np.to_frame(arr, columns=["a", "b", "c"])
    """
    import pandas as _pd
    return _pd.DataFrame(arr, columns=columns, index=index)


def clip_outliers(
    arr: "ndarray",
    method: str = "iqr",
    factor: float = 1.5,
) -> "ndarray":
    """
    Clip outliers in a 1-D array.

    Parameters
    ----------
    method : 'iqr' (interquartile range) | 'zscore' (3-sigma)
    factor : multiplier for the IQR or z-score threshold

    Examples
    --------
    >>> arr_clean = pycrn.np.clip_outliers(arr, method="iqr")
    """
    if method == "iqr":
        q1, q3 = _np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - factor * iqr, q3 + factor * iqr
    elif method == "zscore":
        mu, sigma = arr.mean(), arr.std()
        lo, hi = mu - factor * sigma, mu + factor * sigma
    else:
        raise ValueError(f"Unknown method '{method}'. Use: 'iqr', 'zscore'")
    return _np.clip(arr, lo, hi)


def train_test_split_np(
    *arrays,
    test_size: float = 0.2,
    random_state: _Optional[int] = 42,
) -> _Tuple:
    """
    Simple train/test split for numpy arrays (no sklearn required).

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = pycrn.np.train_test_split_np(X, y)
    """
    n = len(arrays[0])
    rng = _np.random.default_rng(random_state)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    train_idx, test_idx = idx[:split], idx[split:]
    result = []
    for arr in arrays:
        arr = _np.asarray(arr)
        result.extend([arr[train_idx], arr[test_idx]])
    return tuple(result)


def describe(arr: "ndarray") -> dict:
    """
    Summary statistics for a numpy array.

    Examples
    --------
    >>> stats = pycrn.np.describe(arr)
    """
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "mean": float(_np.mean(arr)),
        "std": float(_np.std(arr)),
        "min": float(_np.min(arr)),
        "max": float(_np.max(arr)),
        "median": float(_np.median(arr)),
        "missing": int(_np.isnan(arr).sum()) if _np.issubdtype(arr.dtype, _np.floating) else 0,
    }
