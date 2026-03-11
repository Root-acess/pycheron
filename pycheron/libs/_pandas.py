"""
pycheron.libs._pandas — pandas + pycheron-specific helpers.

Usage:
    import pycheron as pycrn
    df = pycrn.pd.read_csv("data.csv")          # standard pandas
    df = pycrn.pd.read_smart("data.csv")         # auto-detect format
    df = pycrn.pd.clean(df)                      # auto-clean nulls/dupes
    report = pycrn.pd.profile(df)               # statistical profile
    df_train, df_test = pycrn.pd.split(df, target="label")
"""

from __future__ import annotations

# Re-export everything from pandas so pycrn.pd IS pandas
from pandas import *                            # noqa: F401, F403
import pandas as _pd

# Also expose common names explicitly for IDE autocompletion
from pandas import (                            # noqa: F401
    DataFrame, Series, Index, MultiIndex,
    read_csv, read_parquet, read_json, read_excel, read_sql,
    concat, merge, pivot_table, get_dummies,
    to_datetime, to_numeric, isna, notna,
    Categorical, CategoricalDtype,
    options, set_option,
)

from pathlib import Path as _Path
from typing import Optional as _Optional, Tuple as _Tuple, Union as _Union


# ── pycheron-specific helpers ─────────────────────────────────────────────────

def read_smart(
    path: _Union[str, _Path],
    target: _Optional[str] = None,
    **kwargs,
) -> "DataFrame":
    """
    Load CSV, Parquet, JSON or Excel — auto-detects format from extension.

    Parameters
    ----------
    path   : file path (any supported format)
    target : if given, raises if that column is missing
    **kwargs : passed to the underlying pandas reader

    Examples
    --------
    >>> df = pycrn.pd.read_smart("data.csv")
    >>> df = pycrn.pd.read_smart("data.parquet")
    """
    ext = _Path(path).suffix.lower()
    readers = {
        ".csv":     _pd.read_csv,
        ".parquet": _pd.read_parquet,
        ".json":    _pd.read_json,
        ".xlsx":    _pd.read_excel,
        ".xls":     _pd.read_excel,
    }
    if ext not in readers:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {list(readers)}"
        )
    df = readers[ext](path, **kwargs)
    if target and target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"Available: {list(df.columns)}"
        )
    return df


def clean(
    df: "DataFrame",
    *,
    drop_duplicates: bool = True,
    fill_numeric: str = "median",
    fill_categorical: str = "mode",
    drop_missing_threshold: float = 0.7,
    clip_outliers: bool = False,
    verbose: bool = True,
) -> "DataFrame":
    """
    Auto-clean a DataFrame: remove duplicates, fill nulls, drop high-missing cols.

    Parameters
    ----------
    df                     : input DataFrame
    drop_duplicates        : remove duplicate rows (default True)
    fill_numeric           : 'median', 'mean', or 'zero' for numeric nulls
    fill_categorical       : 'mode' or 'unknown' for categorical nulls
    drop_missing_threshold : drop columns with > this fraction missing (default 0.7)
    clip_outliers          : clip numeric outliers to IQR bounds (default False)
    verbose                : print summary of changes

    Returns
    -------
    Cleaned DataFrame (copy, original unchanged).

    Examples
    --------
    >>> df_clean = pycrn.pd.clean(df)
    >>> df_clean = pycrn.pd.clean(df, clip_outliers=True, drop_missing_threshold=0.5)
    """
    from pycheron.data.cleaner import AutoCleaner
    return AutoCleaner(
        drop_duplicates=drop_duplicates,
        fill_numeric=fill_numeric,
        fill_categorical=fill_categorical,
        drop_missing_threshold=drop_missing_threshold,
        clip_outliers=clip_outliers,
        verbose=verbose,
    ).fit_transform(df)


def profile(df: "DataFrame", target: _Optional[str] = None) -> dict:
    """
    Return a quick statistical profile of a DataFrame.

    Examples
    --------
    >>> report = pycrn.pd.profile(df, target="label")
    >>> print(report)
    """
    from pycheron.data.profiler import QuickProfiler
    return QuickProfiler().run(df, target=target)


def split(
    df: "DataFrame",
    target: _Optional[str] = None,
    *,
    test_size: float = 0.2,
    val_size: float = 0.0,
    stratify: bool = True,
    random_state: int = 42,
) -> _Tuple:
    """
    Smart train / (val) / test split with optional stratification.

    Parameters
    ----------
    df           : full DataFrame
    target       : target column for stratification
    test_size    : fraction for test set (default 0.2)
    val_size     : fraction for validation set (default 0.0 = no val set)
    stratify     : use stratified split for classification (default True)
    random_state : random seed

    Returns
    -------
    (df_train, df_test) or (df_train, df_val, df_test) if val_size > 0

    Examples
    --------
    >>> train, test = pycrn.pd.split(df, target="label")
    >>> train, val, test = pycrn.pd.split(df, target="label", val_size=0.1)
    """
    from pycheron.data.splitter import SmartSplitter
    return SmartSplitter(
        test_size=test_size,
        val_size=val_size,
        stratify=stratify,
        random_state=random_state,
    ).split(df, target=target)


def summary(df: "DataFrame") -> "DataFrame":
    """
    Extended describe() — includes dtype, missing %, skewness, cardinality.

    Examples
    --------
    >>> pycrn.pd.summary(df)
    """
    result = []
    for col in df.columns:
        s = df[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing_%": round(s.isnull().mean() * 100, 2),
            "unique": s.nunique(),
            "cardinality_%": round(s.nunique() / max(len(s), 1) * 100, 2),
        }
        if _pd.api.types.is_numeric_dtype(s):
            row.update({
                "mean": round(s.mean(), 4),
                "std": round(s.std(), 4),
                "min": s.min(),
                "max": s.max(),
                "skew": round(s.skew(), 4),
            })
        result.append(row)
    return _pd.DataFrame(result).set_index("column")
