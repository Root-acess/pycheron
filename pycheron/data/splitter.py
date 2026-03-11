"""
pycheron.data.splitter — Smart train/val/test splitting.
"""

from __future__ import annotations

from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from pycheron.utils.logging import get_logger

logger = get_logger(__name__)


class SmartSplitter:
    """
    Intelligent train / (val) / test splitter.

    Automatically uses stratified splitting for classification targets,
    and plain random split for regression targets.

    Parameters
    ----------
    test_size    : fraction held out for test (default 0.2)
    val_size     : fraction held out for validation (default 0.0 = disabled)
    stratify     : use stratified split when target is categorical
    random_state : reproducibility seed

    Examples
    --------
    >>> splitter = SmartSplitter(test_size=0.2, val_size=0.1)
    >>> train, val, test = splitter.split(df, target="label")
    """

    def __init__(
        self,
        test_size: float = 0.2,
        val_size: float = 0.0,
        stratify: bool = True,
        random_state: int = 42,
    ):
        self.test_size = test_size
        self.val_size = val_size
        self.stratify = stratify
        self.random_state = random_state

    def split(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> Tuple:
        """
        Split df into (train, test) or (train, val, test).

        Returns
        -------
        tuple of DataFrames
        """
        stratify_col = None
        if self.stratify and target and target in df.columns:
            y = df[target]
            # Only stratify if target is categorical / low-cardinality
            if y.dtype == object or y.nunique() <= 20:
                stratify_col = y

        # First split: train+val vs test
        df_trainval, df_test = train_test_split(
            df,
            test_size=self.test_size,
            stratify=stratify_col,
            random_state=self.random_state,
        )

        logger.info(
            f"Split: train={len(df_trainval):,} | test={len(df_test):,}"
        )

        if self.val_size > 0:
            # Adjust val fraction relative to trainval size
            val_fraction = self.val_size / (1.0 - self.test_size)
            stratify_trainval = None
            if stratify_col is not None:
                stratify_trainval = df_trainval[target]

            df_train, df_val = train_test_split(
                df_trainval,
                test_size=val_fraction,
                stratify=stratify_trainval,
                random_state=self.random_state,
            )
            logger.info(
                f"  → train={len(df_train):,} | val={len(df_val):,} | test={len(df_test):,}"
            )
            return df_train, df_val, df_test

        return df_trainval, df_test
