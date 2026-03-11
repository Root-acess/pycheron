"""
pycheron.data.profiler — Statistical profiling of datasets.

DataProfiler  : computes summary statistics used by the preprocessing
                engine and algorithm recommendation system.
DataProfile   : immutable result object holding all profile metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from pycheron.data.loader import DataLoader
from pycheron.utils.logging import get_logger
from pycheron.utils.types import ColumnType

logger = get_logger(__name__)


class DataProfile:
    """
    Statistical snapshot of a dataset, used for intelligent ML decisions.

    Attributes
    ----------
    n_rows          : number of rows
    n_features      : number of feature columns (excludes target)
    n_classes       : number of unique target values (None for regression)
    class_balance   : min/max class ratio; 1.0 = perfectly balanced
    feature_types   : dict mapping ColumnType → list of column names
    missing_rates   : per-column missing value fraction
    skewness        : per numeric column skewness
    target_dtype    : pandas dtype string of the target column
    data_hash       : MD5 fingerprint of the full DataFrame

    Properties
    ----------
    is_large        : True when n_rows > 50,000
    is_wide         : True when n_features > 100
    is_imbalanced   : True when class_balance < 0.2
    has_text        : True when TEXT columns are present
    has_datetime    : True when DATETIME columns are present
    """

    def __init__(
        self,
        n_rows: int,
        n_features: int,
        n_classes: Optional[int],
        class_balance: Optional[float],
        feature_types: Dict,
        missing_rates: Dict[str, float],
        skewness: Dict[str, float],
        target_dtype: str,
        data_hash: str,
    ):
        self.n_rows = n_rows
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_balance = class_balance
        self.feature_types = feature_types
        self.missing_rates = missing_rates
        self.skewness = skewness
        self.target_dtype = target_dtype
        self.data_hash = data_hash

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_large(self) -> bool:
        """True when dataset has more than 50,000 rows."""
        return self.n_rows > 50_000

    @property
    def is_wide(self) -> bool:
        """True when dataset has more than 100 features."""
        return self.n_features > 100

    @property
    def is_imbalanced(self) -> bool:
        """True when minority class is less than 20 % of the majority class."""
        return self.class_balance is not None and self.class_balance < 0.2

    @property
    def has_text(self) -> bool:
        """True when at least one TEXT column is present."""
        return bool(self.feature_types.get(ColumnType.TEXT))

    @property
    def has_datetime(self) -> bool:
        """True when at least one DATETIME column is present."""
        return bool(self.feature_types.get(ColumnType.DATETIME))

    def __repr__(self) -> str:
        return (
            f"DataProfile("
            f"rows={self.n_rows:,}, features={self.n_features}, "
            f"classes={self.n_classes}, balance={self.class_balance})"
        )


class DataProfiler:
    """
    Generates a DataProfile from a validated DataFrame.

    The profile drives two key downstream decisions:
      1. Preprocessing strategy (scaler type, encoding method)
      2. Algorithm recommendation scoring

    Examples
    --------
    >>> profiler = DataProfiler()
    >>> profile = profiler.profile(df, target="label", schema=schema)
    >>> profile.is_large      # False
    >>> profile.has_text      # False
    """

    def profile(
        self,
        df: pd.DataFrame,
        target: str,
        schema: Dict,
    ) -> DataProfile:
        """
        Compute a full statistical profile of the dataset.

        Parameters
        ----------
        df     : the raw DataFrame (before preprocessing)
        target : name of the target column
        schema : output of SchemaDetector.detect()

        Returns
        -------
        DataProfile
        """
        n_rows, n_cols = df.shape

        # Group columns by type
        feature_types: Dict = {
            ctype: [
                c for c, t in schema.items()
                if t == ctype and c != target
            ]
            for ctype in ColumnType
        }

        # Target statistics
        y = df[target]
        n_unique_target = int(y.nunique())
        n_classes: Optional[int] = (
            n_unique_target if (y.dtype == object or n_unique_target < 50)
            else None
        )

        class_balance: Optional[float] = None
        if n_classes and n_classes <= 20:
            counts = y.value_counts(normalize=True)
            class_balance = float(counts.min() / counts.max())

        # Per-column skewness for numeric features
        numeric_cols: List[str] = feature_types.get(ColumnType.NUMERIC, [])
        skewness: Dict[str, float] = {}
        if numeric_cols:
            skewness = df[numeric_cols].skew().to_dict()

        return DataProfile(
            n_rows=n_rows,
            n_features=n_cols - 1,
            n_classes=n_classes,
            class_balance=class_balance,
            feature_types=feature_types,
            missing_rates=df.drop(columns=[target]).isnull().mean().to_dict(),
            skewness=skewness,
            target_dtype=str(y.dtype),
            data_hash=DataLoader.hash(df),
        )
