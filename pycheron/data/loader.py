"""
pycheron.data.loader — Unified dataset loading from multiple sources.

Supports: CSV, Parquet, JSON, Excel (.xlsx/.xls), or a pd.DataFrame passed
directly in memory.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd

from pycheron.utils.logging import get_logger
from pycheron.utils.types import DataSource

logger = get_logger(__name__)


class DataLoader:
    """
    Unified interface for loading tabular data from multiple sources.

    Supported formats
    -----------------
    - CSV         (.csv)
    - Parquet     (.parquet)
    - JSON        (.json)
    - Excel       (.xlsx, .xls)
    - pd.DataFrame (in-memory, returned as copy)

    Examples
    --------
    >>> loader = DataLoader()
    >>> df = loader.load("data.csv", target="label")
    >>> df = loader.load(my_dataframe)
    """

    SUPPORTED_EXTENSIONS = {
        ".csv":     "_read_csv",
        ".parquet": "_read_parquet",
        ".json":    "_read_json",
        ".xlsx":    "_read_excel",
        ".xls":     "_read_excel",
    }

    def load(
        self,
        source: DataSource,
        target: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load data and optionally validate that a target column exists.

        Parameters
        ----------
        source : str | Path | pd.DataFrame
            File path or in-memory DataFrame.
        target : str, optional
            If supplied, raises ValueError when the column is missing.

        Returns
        -------
        pd.DataFrame
        """
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Data file not found: {path}")
            ext = path.suffix.lower()
            reader_name = self.SUPPORTED_EXTENSIONS.get(ext)
            if reader_name is None:
                raise ValueError(
                    f"Unsupported file type '{ext}'. "
                    f"Supported: {list(self.SUPPORTED_EXTENSIONS)}"
                )
            df = getattr(self, reader_name)(path)
            logger.info(
                f"Loaded {len(df):,} rows × {len(df.columns)} columns from {path.name}"
            )

        if target and target not in df.columns:
            raise ValueError(
                f"Target column '{target}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        return df

    # ── Private readers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    @staticmethod
    def _read_parquet(path: Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def _read_json(path: Path) -> pd.DataFrame:
        return pd.read_json(path)

    @staticmethod
    def _read_excel(path: Path) -> pd.DataFrame:
        return pd.read_excel(path)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def hash(df: pd.DataFrame) -> str:
        """
        Compute a deterministic MD5 fingerprint of a DataFrame.

        Used for caching decisions and model manifest versioning.
        """
        raw = pd.util.hash_pandas_object(df, index=True).values
        return hashlib.md5(raw.tobytes()).hexdigest()
