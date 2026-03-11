"""
pycheron.data.profiler — Statistical profiling of datasets.

DataProfiler  : full profiler used internally by the trainer
DataProfile   : result object
QuickProfiler : fast human-readable profile (used by pycrn.data.profile())
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

    @property
    def is_large(self) -> bool:
        return self.n_rows > 50_000

    @property
    def is_wide(self) -> bool:
        return self.n_features > 100

    @property
    def is_imbalanced(self) -> bool:
        return self.class_balance is not None and self.class_balance < 0.2

    @property
    def has_text(self) -> bool:
        return bool(self.feature_types.get(ColumnType.TEXT))

    @property
    def has_datetime(self) -> bool:
        return bool(self.feature_types.get(ColumnType.DATETIME))

    def __repr__(self) -> str:
        return (
            f"DataProfile("
            f"rows={self.n_rows:,}, features={self.n_features}, "
            f"classes={self.n_classes}, balance={self.class_balance})"
        )


class DataProfiler:
    """Full profiler used internally by the Trainer."""

    def profile(
        self,
        df: pd.DataFrame,
        target: str,
        schema: Dict,
    ) -> DataProfile:
        n_rows, n_cols = df.shape

        feature_types: Dict = {
            ctype: [
                c for c, t in schema.items()
                if t == ctype and c != target
            ]
            for ctype in ColumnType
        }

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


class QuickProfiler:
    """
    Fast, human-readable profile for pycrn.data.profile() and pycrn.pd.profile().

    Returns a plain dict with summary stats per column.
    """

    def run(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> dict:
        """
        Compute a quick profile of a DataFrame.

        Returns
        -------
        dict with keys: shape, columns, missing, dtypes, numeric_stats,
                        target_info (if target given)
        """
        result: dict = {
            "shape": df.shape,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "total_missing": int(df.isnull().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "columns": {},
        }

        for col in df.columns:
            s = df[col]
            col_info: dict = {
                "dtype": str(s.dtype),
                "missing": int(s.isnull().sum()),
                "missing_pct": round(s.isnull().mean() * 100, 2),
                "unique": int(s.nunique()),
            }
            if pd.api.types.is_numeric_dtype(s):
                col_info.update({
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "median": float(s.median()),
                    "skew": round(float(s.skew()), 4),
                })
            else:
                top = s.value_counts().head(3).to_dict()
                col_info["top_values"] = top

            result["columns"][col] = col_info

        if target and target in df.columns:
            y = df[target]
            result["target"] = {
                "name": target,
                "dtype": str(y.dtype),
                "unique_values": int(y.nunique()),
                "distribution": y.value_counts().to_dict(),
            }
            if y.nunique() <= 20:
                counts = y.value_counts(normalize=True)
                result["target"]["class_balance"] = round(
                    float(counts.min() / counts.max()), 4
                )

        # Display nicely with rich if available
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            t = Table(title=f"Data Profile  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
            t.add_column("Column", style="cyan")
            t.add_column("Dtype")
            t.add_column("Missing %")
            t.add_column("Unique")
            t.add_column("Stats")
            for col, info in result["columns"].items():
                stats = ""
                if "mean" in info:
                    stats = f"mean={info['mean']:.4g}, std={info['std']:.4g}"
                elif "top_values" in info:
                    top_k = list(info["top_values"].keys())[:2]
                    stats = f"top: {top_k}"
                t.add_row(
                    col,
                    info["dtype"],
                    f"{info['missing_pct']}%",
                    str(info["unique"]),
                    stats,
                )
            console.print(t)
        except ImportError:
            pass

        return result
