"""
pymlkit.data.validator — Column schema detection and data quality validation.

SchemaDetector  : classifies every column into a ColumnType
DataValidator   : checks for missing values, duplicates, constant columns
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from pymlkit.utils.logging import get_logger
from pymlkit.utils.types import ColumnType

logger = get_logger(__name__)


# ── SchemaDetector ────────────────────────────────────────────────────────────

class SchemaDetector:
    """
    Automatically infers the semantic type of each column in a DataFrame.

    Column types
    ------------
    NUMERIC     — continuous numbers (int / float with variation)
    CATEGORICAL — object dtype with manageable unique count
    DATETIME    — dates / timestamps
    TEXT        — long free-text strings or very-high-cardinality object columns
    BOOLEAN     — binary 0/1 or True/False columns
    ID          — numeric column where nearly every value is unique
    CONSTANT    — single unique value — useless for ML
    TARGET      — the prediction target (excluded from feature processing)

    Examples
    --------
    >>> detector = SchemaDetector()
    >>> schema = detector.detect(df, target="label")
    >>> schema["age"]        # ColumnType.NUMERIC
    >>> schema["gender"]     # ColumnType.CATEGORICAL
    """

    HIGH_CARDINALITY_THRESHOLD = 50
    MAX_CATEGORIES = 200

    def detect(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
    ) -> Dict[str, ColumnType]:
        """Return a mapping of column name → ColumnType."""
        schema: Dict[str, ColumnType] = {}
        for col in df.columns:
            if col == target:
                schema[col] = ColumnType.TARGET
            else:
                schema[col] = self._classify_column(df[col])
        return schema

    def _classify_column(self, series: pd.Series) -> ColumnType:
        # Constant — no variance, useless for ML
        if series.nunique(dropna=False) <= 1:
            return ColumnType.CONSTANT

        # Datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME

        # String columns that look like dates
        if series.dtype == object:
            sample = series.dropna().head(100)
            if self._looks_like_datetime(sample):
                return ColumnType.DATETIME

        # Boolean
        if pd.api.types.is_bool_dtype(series):
            return ColumnType.BOOLEAN
        if series.nunique() == 2 and set(series.dropna().unique()).issubset(
            {0, 1, "0", "1", True, False}
        ):
            return ColumnType.BOOLEAN

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            # ID heuristic: nearly all values are unique
            if series.nunique() / max(len(series), 1) > 0.95:
                return ColumnType.ID
            return ColumnType.NUMERIC

        # Object (string) columns
        if series.dtype == object:
            avg_len = series.dropna().astype(str).str.len().mean()
            if avg_len > 50:
                return ColumnType.TEXT
            if series.nunique() > self.MAX_CATEGORIES:
                return ColumnType.TEXT
            return ColumnType.CATEGORICAL

        return ColumnType.CATEGORICAL

    @staticmethod
    def _looks_like_datetime(sample: pd.Series) -> bool:
        if sample.empty:
            return False
        try:
            pd.to_datetime(sample.head(20), infer_datetime_format=True)
            return True
        except Exception:
            return False


# ── ValidationIssue / ValidationReport ───────────────────────────────────────

class ValidationIssue:
    """A single data quality warning from DataValidator."""

    def __init__(
        self,
        kind: str,
        column: Optional[str],
        value: object,
        message: str,
    ):
        self.kind = kind
        self.column = column
        self.value = value
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationIssue({self.kind!r}, col={self.column!r})"


class ValidationReport:
    """Collection of ValidationIssues from a single validation run."""

    def __init__(self, issues: List[ValidationIssue]):
        self.issues = issues

    @property
    def has_issues(self) -> bool:
        return bool(self.issues)

    def columns_to_drop(self) -> List[str]:
        """Return columns that should be dropped before training."""
        return [
            i.column
            for i in self.issues
            if i.kind in ("high_missing", "constant") and i.column
        ]

    def __repr__(self) -> str:
        return f"ValidationReport({len(self.issues)} issues)"


# ── DataValidator ─────────────────────────────────────────────────────────────

class DataValidator:
    """
    Validates a DataFrame for data-quality issues and emits warnings.

    Checks performed
    ----------------
    - Columns with > 30 % missing values  → WARNING
    - Columns with > 70 % missing values  → flagged for dropping
    - Duplicate rows
    - Constant columns (will be dropped)
    - Missing values in the target column (rows will be dropped)

    Notes
    -----
    DataValidator never raises by default — issues are surfaced as logged
    warnings and recorded in the returned ValidationReport.

    Examples
    --------
    >>> validator = DataValidator()
    >>> report = validator.validate(df, target="label", schema=schema)
    >>> report.columns_to_drop()
    ['mostly_null_col', 'constant_col']
    """

    MISSING_WARN_THRESHOLD = 0.30
    MISSING_DROP_THRESHOLD = 0.70

    def validate(
        self,
        df: pd.DataFrame,
        target: str,
        schema: Dict[str, ColumnType],
    ) -> ValidationReport:
        """
        Run all validation checks and return a ValidationReport.

        Parameters
        ----------
        df     : DataFrame to validate
        target : name of the target column
        schema : output of SchemaDetector.detect()
        """
        issues: List[ValidationIssue] = []

        # Missing value rates
        missing_rates = df.isnull().mean()
        for col, rate in missing_rates.items():
            if col == target:
                continue
            if rate > self.MISSING_DROP_THRESHOLD:
                issues.append(ValidationIssue(
                    "high_missing", col, rate,
                    f"Column '{col}' has {rate:.0%} missing — will be dropped",
                ))
            elif rate > self.MISSING_WARN_THRESHOLD:
                issues.append(ValidationIssue(
                    "missing", col, rate,
                    f"Column '{col}' has {rate:.0%} missing values",
                ))

        # Duplicate rows
        n_dupes = int(df.duplicated().sum())
        if n_dupes > 0:
            issues.append(ValidationIssue(
                "duplicates", None, n_dupes,
                f"{n_dupes:,} duplicate rows detected",
            ))

        # Constant columns
        for col, ctype in schema.items():
            if ctype == ColumnType.CONSTANT and col != target:
                issues.append(ValidationIssue(
                    "constant", col, None,
                    f"Column '{col}' is constant — will be dropped",
                ))

        # Target missing values
        target_missing = int(df[target].isnull().sum())
        if target_missing > 0:
            issues.append(ValidationIssue(
                "target_missing", target, target_missing,
                f"Target '{target}' has {target_missing} missing values — rows will be dropped",
            ))

        for issue in issues:
            logger.warning(issue.message)

        return ValidationReport(issues)
