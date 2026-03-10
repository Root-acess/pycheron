"""
pymlkit.preprocess.encoder — Categorical and datetime encoding strategies.

Functions
---------
detect_ordinal(series) → Optional[List[str]]
    Checks whether a categorical column follows a known ordinal pattern.

build_categorical_transformer(col, df) → (name, transformer, [col])
    Selects the right sklearn encoder for a single categorical column.

extract_datetime_features(df, col) → pd.DataFrame
    Expands a datetime column into numeric temporal features.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# ── Ordinal pattern library ───────────────────────────────────────────────────

ORDINAL_PATTERNS: List[List[str]] = [
    ["low", "medium", "high"],
    ["small", "medium", "large"],
    ["none", "low", "medium", "high", "very high"],
    ["poor", "fair", "good", "excellent"],
    ["never", "rarely", "sometimes", "often", "always"],
    ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
    ["1", "2", "3", "4", "5"],
    ["very low", "low", "medium", "high", "very high"],
    ["xs", "s", "m", "l", "xl"],
    ["bronze", "silver", "gold", "platinum"],
]


def detect_ordinal(series: pd.Series) -> Optional[List[str]]:
    """
    Return the matched ordinal category order if the column follows a known
    ordinal pattern, otherwise return None.

    Parameters
    ----------
    series : pd.Series
        Categorical string column (NaNs are ignored).

    Returns
    -------
    List[str] with the ordered categories, or None.
    """
    vals = set(series.dropna().astype(str).str.lower().unique())
    for pattern in ORDINAL_PATTERNS:
        if vals and vals.issubset(set(pattern)):
            # Return only the values present, in the correct order
            return [v for v in pattern if v in vals]
    return None


# ── Categorical encoder builder ───────────────────────────────────────────────

def build_categorical_transformer(
    col: str,
    series: pd.Series,
) -> Tuple[str, SkPipeline, List[str]]:
    """
    Select and construct the appropriate categorical encoder for one column.

    Rules
    -----
    1. Ordinal pattern detected  → OrdinalEncoder with explicit category order
    2. ≤ 15 unique values        → OneHotEncoder (drop='if_binary')
    3. > 15 unique values        → OrdinalEncoder (arbitrary hash order)

    Parameters
    ----------
    col    : column name
    series : the column data (used to determine cardinality and ordinal-ness)

    Returns
    -------
    (transformer_name, sklearn_pipeline, [col])
    """
    n_unique = series.nunique()
    ordinal_order: Optional[List[str]] = None

    if series.dtype == object:
        ordinal_order = detect_ordinal(series.dropna().astype(str))

    if ordinal_order:
        enc = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[ordinal_order],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ])
        return (f"ordinal_{col}", enc, [col])

    if n_unique <= 15:
        enc = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop="if_binary",
            )),
        ])
        return (f"ohe_{col}", enc, [col])

    # High-cardinality fallback: ordinal hash encoding
    enc = SkPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    return (f"ord_hc_{col}", enc, [col])


# ── Datetime feature extractor ────────────────────────────────────────────────

def extract_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Expand a datetime column into a set of numeric temporal features.

    Features produced
    -----------------
    year, month, dayofweek, dayofyear, quarter, is_weekend,
    month_sin, month_cos  (cyclical encoding to preserve periodicity)

    Parameters
    ----------
    df  : DataFrame containing the datetime column
    col : name of the datetime column

    Returns
    -------
    pd.DataFrame with 8 new numeric columns (original column excluded).
    """
    dt = pd.to_datetime(df[col], errors="coerce")
    features = pd.DataFrame(index=df.index)
    features[f"{col}_year"]       = dt.dt.year
    features[f"{col}_month"]      = dt.dt.month
    features[f"{col}_dayofweek"]  = dt.dt.dayofweek
    features[f"{col}_dayofyear"]  = dt.dt.dayofyear
    features[f"{col}_quarter"]    = dt.dt.quarter
    features[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    # Cyclical encoding — preserves the circular nature of months
    features[f"{col}_month_sin"]  = np.sin(2 * np.pi * dt.dt.month / 12)
    features[f"{col}_month_cos"]  = np.cos(2 * np.pi * dt.dt.month / 12)
    return features
