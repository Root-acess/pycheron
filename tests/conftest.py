"""
tests/conftest.py — Shared pytest fixtures for all test modules.

Available everywhere without importing:
  clf_df    — 200-row binary classification DataFrame
  reg_df    — 200-row regression DataFrame
  clf_csv   — clf_df saved to a temp CSV file
  model_dir — tmp_path subdirectory for model saving tests
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clf_df():
    """Small binary classification dataset (200 rows)."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "age":    rng.randint(18, 80, n).astype(float),
        "income": rng.exponential(40_000, n),
        "gender": rng.choice(["M", "F"], n),
        "score":  rng.randn(n),
        "label":  rng.choice([0, 1], n),
    })


@pytest.fixture
def reg_df():
    """Small regression dataset (200 rows)."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "x1":  rng.randn(n),
        "x2":  rng.randn(n),
        "cat": rng.choice(["A", "B", "C"], n),
        "y":   rng.randn(n) * 10 + 50,
    })


@pytest.fixture
def clf_csv(clf_df, tmp_path):
    """clf_df serialised to a temp CSV file."""
    p = tmp_path / "clf.csv"
    clf_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def model_dir(tmp_path):
    """Empty temp directory suitable for model.save()."""
    return str(tmp_path / "saved_model")
