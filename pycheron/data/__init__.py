"""
pycheron.data — Data loading, cleaning, profiling, validation, splitting.

Usage:
    import pycheron as pycrn

    df = pycrn.data.load("data.csv")
    df = pycrn.data.clean(df)
    report = pycrn.data.profile(df, target="label")
    train, test = pycrn.data.split(df, target="label")
    issues = pycrn.data.validate(df, target="label")
"""

from __future__ import annotations

from pycheron.data.loader import DataLoader
from pycheron.data.cleaner import AutoCleaner
from pycheron.data.splitter import SmartSplitter
from pycheron.data.profiler import DataProfiler, DataProfile, QuickProfiler
from pycheron.data.validator import SchemaDetector, DataValidator, ValidationReport

import pandas as _pd
from pathlib import Path as _Path
from typing import Optional as _Optional, Union as _Union, Tuple as _Tuple

_loader = DataLoader()


def load(
    path: _Union[str, _Path, _pd.DataFrame],
    target: _Optional[str] = None,
) -> _pd.DataFrame:
    """
    Load data from CSV, Parquet, JSON, Excel, or DataFrame.

    Examples
    --------
    >>> df = pycrn.data.load("data.csv")
    >>> df = pycrn.data.load("data.parquet", target="label")
    """
    return _loader.load(path, target=target)


def clean(
    df: _pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    fill_numeric: str = "median",
    fill_categorical: str = "mode",
    drop_missing_threshold: float = 0.7,
    clip_outliers: bool = False,
    verbose: bool = True,
) -> _pd.DataFrame:
    """
    Auto-clean a DataFrame: drop dupes, fill nulls, remove high-missing columns.

    Examples
    --------
    >>> df_clean = pycrn.data.clean(df)
    """
    return AutoCleaner(
        drop_duplicates=drop_duplicates,
        fill_numeric=fill_numeric,
        fill_categorical=fill_categorical,
        drop_missing_threshold=drop_missing_threshold,
        clip_outliers=clip_outliers,
        verbose=verbose,
    ).fit_transform(df)


def profile(
    df: _pd.DataFrame,
    target: _Optional[str] = None,
) -> dict:
    """
    Statistical profile of a DataFrame.

    Examples
    --------
    >>> report = pycrn.data.profile(df, target="label")
    """
    return QuickProfiler().run(df, target=target)


def validate(
    df: _pd.DataFrame,
    target: str,
) -> "ValidationReport":
    """
    Validate data quality and return a report of issues found.

    Examples
    --------
    >>> report = pycrn.data.validate(df, target="label")
    >>> print(report)
    """
    schema = SchemaDetector().detect(df, target=target)
    return DataValidator().validate(df, target=target, schema=schema)


def split(
    df: _pd.DataFrame,
    target: _Optional[str] = None,
    *,
    test_size: float = 0.2,
    val_size: float = 0.0,
    stratify: bool = True,
    random_state: int = 42,
) -> _Tuple:
    """
    Smart train/val/test split.

    Returns (train, test) or (train, val, test) if val_size > 0.

    Examples
    --------
    >>> train, test = pycrn.data.split(df, target="label")
    >>> train, val, test = pycrn.data.split(df, target="label", val_size=0.1)
    """
    return SmartSplitter(
        test_size=test_size,
        val_size=val_size,
        stratify=stratify,
        random_state=random_state,
    ).split(df, target=target)


__all__ = [
    "load", "clean", "profile", "validate", "split",
    "DataLoader", "AutoCleaner", "SmartSplitter",
    "DataProfiler", "DataProfile", "QuickProfiler",
    "SchemaDetector", "DataValidator", "ValidationReport",
]
