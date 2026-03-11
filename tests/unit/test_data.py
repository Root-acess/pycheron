"""
tests/unit/test_data.py — Unit tests for pycheron.data.*

Covers:
  DataLoader  (loader.py)
  SchemaDetector, DataValidator, ValidationReport  (validator.py)
  DataProfiler, DataProfile  (profiler.py)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pycheron.data.loader import DataLoader
from pycheron.data.validator import SchemaDetector, DataValidator, ValidationReport
from pycheron.data.profiler import DataProfiler, DataProfile
from pycheron.utils.types import ColumnType


# ── DataLoader ────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_dataframe_returns_copy(self, clf_df):
        loader = DataLoader()
        df = loader.load(clf_df)
        assert df.shape == clf_df.shape
        df["__test__"] = 0
        assert "__test__" not in clf_df.columns  # must be a copy

    def test_load_csv(self, clf_csv):
        df = DataLoader().load(clf_csv)
        assert len(df) > 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            DataLoader().load("nonexistent_file.csv")

    def test_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            DataLoader().load(str(f))

    def test_bad_target_raises(self, clf_df):
        with pytest.raises(ValueError, match="not found"):
            DataLoader().load(clf_df, target="nonexistent_col")

    def test_hash_is_deterministic(self, clf_df):
        h1 = DataLoader.hash(clf_df)
        h2 = DataLoader.hash(clf_df.copy())
        assert h1 == h2

    def test_hash_changes_on_modification(self, clf_df):
        h1 = DataLoader.hash(clf_df)
        df2 = clf_df.copy()
        df2.loc[0, "age"] = -999.0
        h2 = DataLoader.hash(df2)
        assert h1 != h2


# ── SchemaDetector ────────────────────────────────────────────────────────────

class TestSchemaDetector:
    def test_numeric_columns(self, clf_df):
        schema = SchemaDetector().detect(clf_df, target="label")
        assert schema["age"]    == ColumnType.NUMERIC
        assert schema["income"] == ColumnType.NUMERIC

    def test_categorical_column(self, clf_df):
        schema = SchemaDetector().detect(clf_df, target="label")
        assert schema["gender"] == ColumnType.CATEGORICAL

    def test_target_column(self, clf_df):
        schema = SchemaDetector().detect(clf_df, target="label")
        assert schema["label"] == ColumnType.TARGET

    def test_constant_column(self):
        df = pd.DataFrame({"const": [1] * 50, "y": [0, 1] * 25})
        schema = SchemaDetector().detect(df, target="y")
        assert schema["const"] == ColumnType.CONSTANT

    def test_datetime_column(self):
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=50),
            "y":    range(50),
        })
        schema = SchemaDetector().detect(df, target="y")
        assert schema["date"] == ColumnType.DATETIME

    def test_boolean_column(self):
        df = pd.DataFrame({"flag": [0, 1, 0, 1, 1], "y": [0, 1, 0, 1, 0]})
        schema = SchemaDetector().detect(df, target="y")
        assert schema["flag"] == ColumnType.BOOLEAN

    def test_text_column(self):
        long_text = ["This is a very long free-text description. " * 3] * 50
        df = pd.DataFrame({"review": long_text, "y": [0, 1] * 25})
        schema = SchemaDetector().detect(df, target="y")
        assert schema["review"] == ColumnType.TEXT


# ── DataValidator ─────────────────────────────────────────────────────────────

class TestDataValidator:
    def _schema(self, df, target):
        return SchemaDetector().detect(df, target=target)

    def test_detects_high_missing(self):
        df = pd.DataFrame({
            "good":        range(100),
            "mostly_null": [None] * 80 + list(range(20)),
            "y":           [0, 1] * 50,
        })
        report = DataValidator().validate(df, "y", self._schema(df, "y"))
        assert "mostly_null" in report.columns_to_drop()

    def test_clean_data_has_no_drops(self, clf_df):
        report = DataValidator().validate(clf_df, "label", self._schema(clf_df, "label"))
        assert len(report.columns_to_drop()) == 0

    def test_constant_column_flagged_for_drop(self):
        df = pd.DataFrame({"const": [1] * 50, "x": range(50), "y": [0, 1] * 25})
        report = DataValidator().validate(df, "y", self._schema(df, "y"))
        assert "const" in report.columns_to_drop()

    def test_duplicate_rows_detected(self):
        df = pd.DataFrame({"a": [1, 1, 2], "y": [0, 0, 1]})
        report = DataValidator().validate(df, "y", self._schema(df, "y"))
        dupe_issues = [i for i in report.issues if i.kind == "duplicates"]
        assert len(dupe_issues) == 1

    def test_target_missing_detected(self):
        df = pd.DataFrame({"x": range(10), "y": [0, None] * 5})
        report = DataValidator().validate(df, "y", self._schema(df, "y"))
        target_issues = [i for i in report.issues if i.kind == "target_missing"]
        assert len(target_issues) == 1


# ── DataProfiler ─────────────────────────────────────────────────────────────

class TestDataProfiler:
    def _run(self, df, target):
        schema = SchemaDetector().detect(df, target=target)
        return DataProfiler().profile(df, target=target, schema=schema)

    def test_n_rows_and_features(self, clf_df):
        profile = self._run(clf_df, "label")
        assert profile.n_rows == len(clf_df)
        assert profile.n_features == len(clf_df.columns) - 1

    def test_n_classes_binary(self, clf_df):
        profile = self._run(clf_df, "label")
        assert profile.n_classes == 2

    def test_data_hash_populated(self, clf_df):
        profile = self._run(clf_df, "label")
        assert len(profile.data_hash) == 32  # MD5 hex

    def test_is_large_false_for_small(self, clf_df):
        assert not self._run(clf_df, "label").is_large

    def test_is_imbalanced_false_for_balanced(self, clf_df):
        # clf_df is roughly balanced
        assert not self._run(clf_df, "label").is_imbalanced

    def test_skewness_populated(self, clf_df):
        profile = self._run(clf_df, "label")
        assert isinstance(profile.skewness, dict)
