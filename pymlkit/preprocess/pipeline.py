"""
pymlkit.preprocess.pipeline — Assembles and fits the full feature pipeline.

Preprocessor orchestrates:
  1. Drop columns flagged by DataValidator (constant / too-many-missing)
  2. Separate target from features
  3. Encode the target (LabelEncoder for classification)
  4. Build a sklearn ColumnTransformer using encoder.py and scaler.py
  5. Return (X, y) as numpy arrays ready for training
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import LabelEncoder

from pymlkit.data.profiler import DataProfile
from pymlkit.data.validator import ValidationReport
from pymlkit.preprocess.encoder import (
    build_categorical_transformer,
    extract_datetime_features,
)
from pymlkit.preprocess.scaler import choose_scaler, is_tree_based
from pymlkit.utils.logging import get_logger
from pymlkit.utils.types import ColumnType, TaskType

logger = get_logger(__name__)


class Preprocessor:
    """
    Automatically builds and fits a sklearn ColumnTransformer pipeline.

    The transformer is constructed once during fit_transform() and reused
    in transform() for inference — ensuring train/test consistency.

    Parameters
    ----------
    algorithm : str
        Name of the training algorithm (affects scaler selection).
    task : TaskType
        ML task type (affects target encoding).

    Examples
    --------
    >>> prep = Preprocessor(algorithm="random_forest", task=TaskType.BINARY_CLASSIFICATION)
    >>> X, y = prep.fit_transform(df, "label", schema, profile, validation)
    >>> X_new = prep.transform(new_df)
    """

    def __init__(
        self,
        algorithm: str = "random_forest",
        task: TaskType = TaskType.BINARY_CLASSIFICATION,
    ):
        self.algorithm = algorithm
        self.task = task
        self._transformer: Optional[ColumnTransformer] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._feature_names_out: Optional[List[str]] = None
        self._columns_dropped: List[str] = []
        self._schema: Dict[str, ColumnType] = {}

    # ── Public interface ──────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
        target: str,
        schema: Dict[str, ColumnType],
        profile: DataProfile,
        validation: ValidationReport,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessing pipeline and transform the entire dataset.

        Parameters
        ----------
        df         : raw DataFrame (includes target column)
        target     : name of the target column
        schema     : output of SchemaDetector.detect()
        profile    : output of DataProfiler.profile()
        validation : output of DataValidator.validate()

        Returns
        -------
        (X, y) as (np.ndarray, np.ndarray)
        """
        self._schema = schema

        # 1. Drop columns flagged by the validator
        cols_to_drop = validation.columns_to_drop()
        df = df.drop(columns=cols_to_drop, errors="ignore")
        self._columns_dropped = cols_to_drop
        if cols_to_drop:
            logger.info(f"Dropped {len(cols_to_drop)} column(s): {cols_to_drop}")

        # 2. Separate target — also drop rows where target is null
        y_raw = df[target].dropna()
        df = df.loc[y_raw.index].drop(columns=[target])

        # 3. Encode target
        y = self._encode_target(y_raw)

        # 4. Build feature transformer and fit+transform
        X = self._build_and_fit(df, schema, profile)

        logger.info(f"Preprocessing complete: X={X.shape}, y={y.shape}")
        return X, y

    def transform(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> np.ndarray:
        """
        Transform new data using the already-fitted pipeline.

        Parameters
        ----------
        df     : new DataFrame (may include target column — it is ignored)
        target : optional target column name to exclude

        Returns
        -------
        np.ndarray
        """
        if self._transformer is None:
            raise RuntimeError(
                "Preprocessor not fitted. Call fit_transform() before transform()."
            )
        drop_cols = list(self._columns_dropped)
        if target:
            drop_cols.append(target)
        df = df.drop(columns=drop_cols, errors="ignore")
        return self._transformer.transform(df)

    @property
    def feature_names(self) -> List[str]:
        """Feature names after encoding (mirrors ColumnTransformer output)."""
        return self._feature_names_out or []

    # ── Target encoding ───────────────────────────────────────────────────────

    def _encode_target(self, y: pd.Series) -> np.ndarray:
        if self.task.is_classification and y.dtype == object:
            self._label_encoder = LabelEncoder()
            return self._label_encoder.fit_transform(y)
        return y.values

    def decode_target(self, y: np.ndarray) -> np.ndarray:
        """Reverse label encoding to restore original class strings."""
        if self._label_encoder is not None:
            return self._label_encoder.inverse_transform(y)
        return y

    # ── ColumnTransformer builder ─────────────────────────────────────────────

    def _build_and_fit(
        self,
        df: pd.DataFrame,
        schema: Dict[str, ColumnType],
        profile: DataProfile,
    ) -> np.ndarray:
        transformers = []

        # ── Numeric ───────────────────────────────────────────────────────────
        numeric_cols = [
            c for c, t in schema.items()
            if t == ColumnType.NUMERIC and c in df.columns
        ]
        if numeric_cols:
            scaler = choose_scaler(numeric_cols, profile.skewness, self.algorithm)
            steps = [("imputer", SimpleImputer(strategy="median"))]
            if scaler is not None:
                steps.append(("scaler", scaler))
            transformers.append(("numeric", SkPipeline(steps), numeric_cols))

        # ── Boolean ───────────────────────────────────────────────────────────
        bool_cols = [
            c for c, t in schema.items()
            if t == ColumnType.BOOLEAN and c in df.columns
        ]
        if bool_cols:
            df[bool_cols] = df[bool_cols].astype(int)
            transformers.append(("boolean", "passthrough", bool_cols))

        # ── Categorical ───────────────────────────────────────────────────────
        cat_cols = [
            c for c, t in schema.items()
            if t == ColumnType.CATEGORICAL and c in df.columns
        ]
        for col in cat_cols:
            name, enc, cols = build_categorical_transformer(col, df[col])
            transformers.append((name, enc, cols))

        # ── Datetime ──────────────────────────────────────────────────────────
        dt_cols = [
            c for c, t in schema.items()
            if t == ColumnType.DATETIME and c in df.columns
        ]
        for col in dt_cols:
            dt_features = extract_datetime_features(df, col)
            df = pd.concat([df, dt_features], axis=1)
            new_cols = list(dt_features.columns)
            transformers.append((
                f"dt_{col}",
                SkPipeline([("imputer", SimpleImputer(strategy="median"))]),
                new_cols,
            ))

        # ── Skip non-informative columns ──────────────────────────────────────
        skipped = [
            c for c, t in schema.items()
            if t in (ColumnType.ID, ColumnType.CONSTANT, ColumnType.TEXT)
            and c in df.columns
        ]
        if skipped:
            logger.info(f"Skipping non-informative columns: {skipped}")

        if not transformers:
            raise ValueError(
                "No usable feature columns found after preprocessing. "
                "Check that your DataFrame has feature columns beyond the target."
            )

        self._transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            n_jobs=1,
        )
        X = self._transformer.fit_transform(df)

        # Capture output feature names for explainability
        try:
            self._feature_names_out = (
                self._transformer.get_feature_names_out().tolist()
            )
        except Exception:
            self._feature_names_out = [f"feature_{i}" for i in range(X.shape[1])]

        return X
