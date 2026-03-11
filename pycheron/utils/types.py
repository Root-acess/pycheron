"""
pycheron.utils.types — Shared type definitions and enums.
"""

from __future__ import annotations

from enum import Enum
from typing import Union
from pathlib import Path

import pandas as pd


class TaskType(str, Enum):
    """ML task type detected from the target column."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    UNKNOWN = "unknown"

    @property
    def is_classification(self) -> bool:
        return self in (self.BINARY_CLASSIFICATION, self.MULTICLASS_CLASSIFICATION)

    @property
    def is_regression(self) -> bool:
        return self == self.REGRESSION


class ColumnType(str, Enum):
    """Semantic type of a DataFrame column."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    ID = "id"
    CONSTANT = "constant"
    TARGET = "target"


# Type aliases
DataSource = Union[str, Path, pd.DataFrame]
