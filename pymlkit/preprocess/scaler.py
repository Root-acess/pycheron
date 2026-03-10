"""
pymlkit.preprocess.scaler — Numeric scaling strategy selection.

The right scaler depends on:
  - The target algorithm (tree-based models skip scaling entirely)
  - The distribution of numeric features (skewed data → RobustScaler)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import RobustScaler, StandardScaler

# Algorithms that are inherently scale-invariant (split-based decisions)
TREE_BASED_ALGORITHMS = frozenset({
    "random_forest",
    "random_forest_regressor",
    "gradient_boosting",
    "gradient_boosting_regressor",
    "xgboost",
    "xgboost_regressor",
    "lightgbm",
    "lightgbm_regressor",
    "extra_trees",
    "extra_trees_regressor",
    "decision_tree",
    "decision_tree_regressor",
})

# Skewness magnitude above which a feature is considered "highly skewed"
HIGH_SKEW_THRESHOLD = 2.0

# Proportion of highly-skewed features above which RobustScaler is preferred
ROBUST_PREFERENCE_FRACTION = 0.5


def is_tree_based(algorithm: str) -> bool:
    """Return True if the algorithm does not benefit from feature scaling."""
    return algorithm in TREE_BASED_ALGORITHMS


def choose_scaler(
    numeric_cols: List[str],
    skewness: Dict[str, float],
    algorithm: str,
):
    """
    Select the most appropriate sklearn scaler for numeric features.

    Decision rules
    --------------
    1. Tree-based algorithm → return None (no scaling needed)
    2. Majority of features highly skewed (|skew| > 2) → RobustScaler
    3. Otherwise → StandardScaler

    Parameters
    ----------
    numeric_cols : list of numeric column names
    skewness     : mapping col → skewness score from DataProfile
    algorithm    : name of the training algorithm

    Returns
    -------
    sklearn scaler instance, or None for tree-based algorithms.
    """
    if is_tree_based(algorithm):
        return None

    if not numeric_cols:
        return StandardScaler()

    n_skewed = sum(
        1 for col in numeric_cols
        if abs(skewness.get(col, 0.0)) > HIGH_SKEW_THRESHOLD
    )
    skew_fraction = n_skewed / len(numeric_cols)

    if skew_fraction > ROBUST_PREFERENCE_FRACTION:
        return RobustScaler()

    return StandardScaler()
