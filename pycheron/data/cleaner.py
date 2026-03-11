"""
pycheron.data.cleaner — AutoCleaner for automatic data cleaning.

Handles: duplicates, missing values, high-missing columns, outliers,
         class imbalance detection, and constant columns.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np

from pycheron.utils.logging import get_logger

logger = get_logger(__name__)


class AutoCleaner:
    """
    Automatically cleans a DataFrame before ML training.

    Operations (in order)
    ---------------------
    1. Drop columns with too many missing values
    2. Drop duplicate rows
    3. Fill numeric nulls (median / mean / zero)
    4. Fill categorical nulls (mode / 'unknown')
    5. Optionally clip outliers with IQR method
    6. Report imbalance if target is provided

    Parameters
    ----------
    drop_duplicates        : remove duplicate rows
    fill_numeric           : 'median' | 'mean' | 'zero'
    fill_categorical       : 'mode' | 'unknown'
    drop_missing_threshold : drop columns where missing > this fraction
    clip_outliers          : clip numeric outliers to IQR bounds
    verbose                : log a summary of changes

    Examples
    --------
    >>> cleaner = AutoCleaner()
    >>> df_clean = cleaner.fit_transform(df)
    """

    def __init__(
        self,
        drop_duplicates: bool = True,
        fill_numeric: str = "median",
        fill_categorical: str = "mode",
        drop_missing_threshold: float = 0.7,
        clip_outliers: bool = False,
        verbose: bool = True,
    ):
        self.drop_duplicates = drop_duplicates
        self.fill_numeric = fill_numeric
        self.fill_categorical = fill_categorical
        self.drop_missing_threshold = drop_missing_threshold
        self.clip_outliers = clip_outliers
        self.verbose = verbose

        # Fitted state (for re-use on new data)
        self._fill_values: dict = {}
        self._cols_to_drop: list = []
        self._iqr_bounds: dict = {}
        self._fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fit cleaning parameters on df and return a cleaned copy.

        Parameters
        ----------
        df     : input DataFrame (not modified in place)
        target : target column (excluded from feature cleaning)
        """
        df = df.copy()
        original_shape = df.shape
        changes = []

        # 1. Drop columns with too many missing values
        feature_cols = [c for c in df.columns if c != target]
        missing_rates = df[feature_cols].isnull().mean()
        self._cols_to_drop = missing_rates[
            missing_rates > self.drop_missing_threshold
        ].index.tolist()
        if self._cols_to_drop:
            df = df.drop(columns=self._cols_to_drop)
            changes.append(f"Dropped {len(self._cols_to_drop)} high-missing column(s): {self._cols_to_drop}")

        # 2. Drop duplicates
        if self.drop_duplicates:
            n_before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            n_dupes = n_before - len(df)
            if n_dupes:
                changes.append(f"Removed {n_dupes:,} duplicate rows")

        # 3. Fill nulls per column
        for col in df.columns:
            if col == target:
                continue
            if df[col].isnull().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if self.fill_numeric == "median":
                    fill = df[col].median()
                elif self.fill_numeric == "mean":
                    fill = df[col].mean()
                else:
                    fill = 0
                self._fill_values[col] = fill
                df[col] = df[col].fillna(fill)
                changes.append(f"Filled '{col}' numeric nulls with {self.fill_numeric}={fill:.4g}")
            else:
                if self.fill_numeric == "mode" or self.fill_categorical == "mode":
                    mode = df[col].mode()
                    fill = mode.iloc[0] if len(mode) > 0 else "unknown"
                else:
                    fill = "unknown"
                self._fill_values[col] = fill
                df[col] = df[col].fillna(fill)
                changes.append(f"Filled '{col}' categorical nulls with '{fill}'")

        # 4. Clip outliers
        if self.clip_outliers:
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != target
            ]
            for col in numeric_cols:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                self._iqr_bounds[col] = (lo, hi)
                n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
                if n_clipped:
                    df[col] = df[col].clip(lo, hi)
                    changes.append(f"Clipped {n_clipped} outliers in '{col}' [{lo:.4g}, {hi:.4g}]")

        # 5. Class imbalance warning
        if target and target in df.columns:
            counts = df[target].value_counts(normalize=True)
            balance = counts.min() / counts.max()
            if balance < 0.3:
                logger.warning(
                    f"Class imbalance detected: balance ratio={balance:.2f}. "
                    f"Consider using class_weight='balanced' or resampling."
                )

        self._fitted = True

        if self.verbose:
            final_shape = df.shape
            logger.info(
                f"AutoCleaner: {original_shape} → {final_shape} | "
                f"{len(changes)} operation(s) applied"
            )
            for c in changes:
                logger.info(f"  • {c}")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted cleaning to new data (uses stored fill values)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")
        df = df.copy()
        df = df.drop(columns=self._cols_to_drop, errors="ignore")
        for col, fill in self._fill_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill)
        for col, (lo, hi) in self._iqr_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df
