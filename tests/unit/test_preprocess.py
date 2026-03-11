"""
tests/unit/test_preprocess.py — Unit tests for pycheron.preprocess.*

Covers:
  detect_ordinal     (encoder.py)
  choose_scaler      (scaler.py)
  Preprocessor       (pipeline.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycheron.data.profiler import DataProfiler
from pycheron.data.validator import DataValidator, SchemaDetector
from pycheron.preprocess.encoder import detect_ordinal, extract_datetime_features
from pycheron.preprocess.pipeline import Preprocessor
from pycheron.preprocess.scaler import choose_scaler, is_tree_based
from pycheron.utils.types import TaskType


# ── encoder.py ────────────────────────────────────────────────────────────────

class TestDetectOrdinal:
    def test_recognises_low_medium_high(self):
        s = pd.Series(["low", "medium", "high", "low", "medium"])
        result = detect_ordinal(s)
        assert result == ["low", "medium", "high"]

    def test_returns_none_for_non_ordinal(self):
        s = pd.Series(["cat", "dog", "fish"])
        assert detect_ordinal(s) is None

    def test_case_insensitive(self):
        s = pd.Series(["Low", "Medium", "High"])
        result = detect_ordinal(s)
        assert result is not None

    def test_partial_pattern_matches(self):
        # Only "low" and "high" present — still ordinal
        s = pd.Series(["low", "high", "low"])
        result = detect_ordinal(s)
        assert result == ["low", "high"]


class TestExtractDatetimeFeatures:
    def test_produces_eight_features(self):
        df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10)})
        features = extract_datetime_features(df, "date")
        assert features.shape == (10, 8)

    def test_contains_expected_columns(self):
        df = pd.DataFrame({"ts": pd.date_range("2021-06-01", periods=5)})
        features = extract_datetime_features(df, "ts")
        assert "ts_year" in features.columns
        assert "ts_month_sin" in features.columns
        assert "ts_is_weekend" in features.columns


# ── scaler.py ─────────────────────────────────────────────────────────────────

class TestScaler:
    def test_tree_based_returns_none(self):
        scaler = choose_scaler(["x1", "x2"], {"x1": 0.5, "x2": 0.2}, "random_forest")
        assert scaler is None

    def test_linear_returns_standard_scaler(self):
        from sklearn.preprocessing import StandardScaler
        scaler = choose_scaler(["x1"], {"x1": 0.5}, "ridge")
        assert isinstance(scaler, StandardScaler)

    def test_skewed_data_returns_robust_scaler(self):
        from sklearn.preprocessing import RobustScaler
        # All columns highly skewed
        scaler = choose_scaler(
            ["x1", "x2", "x3"],
            {"x1": 3.0, "x2": 4.0, "x3": 5.0},
            "logistic_regression",
        )
        assert isinstance(scaler, RobustScaler)

    def test_is_tree_based_true(self):
        assert is_tree_based("random_forest")
        assert is_tree_based("xgboost")
        assert is_tree_based("lightgbm")

    def test_is_tree_based_false(self):
        assert not is_tree_based("ridge")
        assert not is_tree_based("logistic_regression")
        assert not is_tree_based("svm")


# ── pipeline.py ───────────────────────────────────────────────────────────────

class TestPreprocessor:
    def _run(self, df, target, algorithm="random_forest"):
        schema     = SchemaDetector().detect(df, target=target)
        validation = DataValidator().validate(df, target=target, schema=schema)
        profile    = DataProfiler().profile(df, target=target, schema=schema)
        prep       = Preprocessor(algorithm=algorithm, task=TaskType.BINARY_CLASSIFICATION)
        X, y = prep.fit_transform(df, target, schema, profile, validation)
        return X, y, prep

    def test_output_shapes_match_input(self, clf_df):
        X, y, _ = self._run(clf_df, "label")
        assert X.shape[0] == len(clf_df)
        assert y.shape[0] == len(clf_df)

    def test_no_nans_in_X(self, clf_df):
        df = clf_df.copy()
        df.loc[:5, "age"] = np.nan  # inject missing values
        X, _, _ = self._run(df, "label")
        assert not np.any(np.isnan(X))

    def test_feature_names_length_matches_X_columns(self, clf_df):
        X, _, prep = self._run(clf_df, "label")
        assert len(prep.feature_names) == X.shape[1]

    def test_transform_matches_fit_transform_columns(self, clf_df):
        X, _, prep = self._run(clf_df, "label")
        X2 = prep.transform(clf_df.drop(columns=["label"]))
        assert X2.shape[1] == X.shape[1]

    def test_datetime_column_expanded(self):
        df = pd.DataFrame({
            "date":  pd.date_range("2020-01-01", periods=50),
            "value": np.random.randn(50),
            "y":     np.random.choice([0, 1], 50),
        })
        X, _, prep = self._run(df, "y")
        # Datetime should produce multiple features — more than just value
        assert X.shape[1] > 1

    def test_constant_column_dropped(self):
        df = pd.DataFrame({
            "useful": np.random.randn(50),
            "const":  [1] * 50,
            "y":      np.random.choice([0, 1], 50),
        })
        X, _, prep = self._run(df, "y")
        assert "const" in prep._columns_dropped

    def test_transform_without_fit_raises(self, clf_df):
        prep = Preprocessor()
        with pytest.raises(RuntimeError, match="not fitted"):
            prep.transform(clf_df)
