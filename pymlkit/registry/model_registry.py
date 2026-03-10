"""
pymlkit.registry — Algorithm registry with auto-recommendation.

All algorithms (built-in + third-party plugins) are registered here.
The registry scores candidates against a DataProfile to recommend the
best algorithm for a given dataset.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
from sklearn.base import BaseEstimator

from pymlkit.utils.logging import get_logger
from pymlkit.utils.types import TaskType

logger = get_logger(__name__)


# ── AlgorithmMeta ─────────────────────────────────────────────────────────────

@dataclass
class AlgorithmMeta:
    """Metadata descriptor for a registered algorithm."""
    name: str
    factory: Callable[..., BaseEstimator]
    tasks: List[str]                       # e.g. ["classification", "regression"]
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_space: Dict[str, Any] = field(default_factory=dict)  # Optuna search space
    requires_scaling: bool = False
    memory_heavy: bool = False
    gpu_supported: bool = False
    description: str = ""

    def build(self, **override_params) -> BaseEstimator:
        """Instantiate the estimator with merged params."""
        params = {**self.default_params, **override_params}
        return self.factory(**params)

    def supports(self, task: TaskType) -> bool:
        if task.is_classification:
            return "classification" in self.tasks
        if task.is_regression:
            return "regression" in self.tasks
        return False


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Singleton registry of all available ML algorithms.

    Usage:
        registry = ModelRegistry.instance()
        registry.recommend(profile, task)
    """

    _instance: Optional["ModelRegistry"] = None

    def __init__(self) -> None:
        self._algorithms: Dict[str, AlgorithmMeta] = {}
        self._register_builtins()

    @classmethod
    def instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, meta: AlgorithmMeta) -> None:
        if meta.name in self._algorithms:
            logger.debug(f"Re-registering algorithm '{meta.name}'")
        self._algorithms[meta.name] = meta
        logger.debug(f"Registered algorithm: {meta.name}")

    def get(self, name: str) -> AlgorithmMeta:
        if name not in self._algorithms:
            available = list(self._algorithms)
            raise ValueError(
                f"Unknown algorithm '{name}'. Available: {available}"
            )
        return self._algorithms[name]

    def list_names(self, task: Optional[TaskType] = None) -> List[str]:
        if task is None:
            return list(self._algorithms)
        return [
            name for name, meta in self._algorithms.items()
            if meta.supports(task)
        ]

    # ── Recommendation ───────────────────────────────────────────────────

    def recommend(
        self,
        profile,
        task: TaskType,
        n: int = 6,
    ) -> List[str]:
        """
        Rank algorithms by suitability for the given dataset profile.

        Scoring heuristics:
        - Large datasets → prefer fast algorithms (LightGBM, RandomForest)
        - Small datasets → linear models competitive
        - Imbalanced → gradient boosting handles better
        - Wide datasets → linear models or trees
        """
        candidates = [
            (name, meta) for name, meta in self._algorithms.items()
            if meta.supports(task)
        ]

        scored = []
        for name, meta in candidates:
            score = self._score_algorithm(meta, profile)
            scored.append((score, name))

        scored.sort(reverse=True)
        top = [name for _, name in scored[:n]]
        logger.info(f"Recommended algorithms: {top}")
        return top

    def _score_algorithm(self, meta: AlgorithmMeta, profile: "DataProfile") -> float:
        score = 0.0

        # Large dataset preference
        if profile.is_large:
            if meta.name in ("lightgbm", "random_forest", "gradient_boosting"):
                score += 2.0
            if meta.name in ("svm", "knn"):
                score -= 2.0

        # Wide dataset preference
        if profile.is_wide:
            if meta.name in ("lasso", "ridge", "logistic_regression"):
                score += 1.5

        # Imbalanced preference
        if profile.is_imbalanced:
            if meta.name in ("gradient_boosting", "xgboost", "lightgbm", "random_forest"):
                score += 1.5

        # Text features → tree ensembles are weaker; gradient boosting better
        if profile.has_text:
            if meta.name in ("gradient_boosting", "xgboost", "lightgbm"):
                score += 1.0

        # Penalize memory-heavy on large data
        if profile.is_large and meta.memory_heavy:
            score -= 1.5

        return score

    # ── Built-in Algorithm Registrations ─────────────────────────────────

    def _register_builtins(self) -> None:
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
            ExtraTreesClassifier, ExtraTreesRegressor,
        )
        from sklearn.linear_model import (
            LogisticRegression, Ridge, Lasso,
            ElasticNet, LinearRegression,
        )
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        builtins = [
            # ── Classification ─────────────────────────────────────────
            AlgorithmMeta(
                name="random_forest",
                factory=lambda **kw: RandomForestClassifier(n_jobs=-1, **kw),
                tasks=["classification"],
                default_params={"n_estimators": 200, "random_state": 42},
                param_space={
                    "n_estimators": ("int", 50, 500),
                    "max_depth": ("int_none", 3, 20),
                    "min_samples_split": ("int", 2, 20),
                    "max_features": ("categorical", ["sqrt", "log2", None]),
                },
                description="Ensemble of decision trees. Robust default choice.",
            ),
            AlgorithmMeta(
                name="gradient_boosting",
                factory=lambda **kw: GradientBoostingClassifier(**kw),
                tasks=["classification"],
                default_params={"n_estimators": 200, "random_state": 42},
                param_space={
                    "n_estimators": ("int", 50, 400),
                    "learning_rate": ("float_log", 0.01, 0.3),
                    "max_depth": ("int", 2, 8),
                    "subsample": ("float", 0.5, 1.0),
                },
                description="Gradient boosted trees. Strong on tabular data.",
            ),
            AlgorithmMeta(
                name="logistic_regression",
                factory=lambda **kw: LogisticRegression(max_iter=1000, n_jobs=-1, **kw),
                tasks=["classification"],
                default_params={"C": 1.0, "random_state": 42},
                requires_scaling=True,
                param_space={
                    "C": ("float_log", 1e-4, 10.0),
                    "solver": ("categorical", ["lbfgs", "saga"]),
                },
                description="Fast, interpretable linear classifier.",
            ),
            AlgorithmMeta(
                name="knn",
                factory=lambda **kw: KNeighborsClassifier(n_jobs=-1, **kw),
                tasks=["classification"],
                default_params={"n_neighbors": 5},
                requires_scaling=True,
                memory_heavy=True,
                param_space={
                    "n_neighbors": ("int", 1, 30),
                    "weights": ("categorical", ["uniform", "distance"]),
                },
                description="K-Nearest Neighbors. Simple, no assumptions.",
            ),
            AlgorithmMeta(
                name="svm",
                factory=lambda **kw: SVC(probability=True, **kw),
                tasks=["classification"],
                default_params={"C": 1.0, "kernel": "rbf", "random_state": 42},
                requires_scaling=True,
                memory_heavy=True,
                param_space={
                    "C": ("float_log", 0.01, 10.0),
                    "kernel": ("categorical", ["rbf", "linear", "poly"]),
                },
                description="Support Vector Machine. Strong on small/medium datasets.",
            ),
            AlgorithmMeta(
                name="decision_tree",
                factory=lambda **kw: DecisionTreeClassifier(**kw),
                tasks=["classification"],
                default_params={"random_state": 42},
                param_space={
                    "max_depth": ("int_none", 2, 20),
                    "min_samples_split": ("int", 2, 20),
                    "criterion": ("categorical", ["gini", "entropy"]),
                },
                description="Single decision tree. Highly interpretable.",
            ),
            AlgorithmMeta(
                name="extra_trees",
                factory=lambda **kw: ExtraTreesClassifier(n_jobs=-1, **kw),
                tasks=["classification"],
                default_params={"n_estimators": 200, "random_state": 42},
                param_space={
                    "n_estimators": ("int", 50, 400),
                    "max_depth": ("int_none", 3, 20),
                },
                description="Extremely Randomized Trees. Faster than RandomForest.",
            ),
            # ── Regression ──────────────────────────────────────────────
            AlgorithmMeta(
                name="random_forest_regressor",
                factory=lambda **kw: RandomForestRegressor(n_jobs=-1, **kw),
                tasks=["regression"],
                default_params={"n_estimators": 200, "random_state": 42},
                param_space={
                    "n_estimators": ("int", 50, 500),
                    "max_depth": ("int_none", 3, 20),
                    "min_samples_split": ("int", 2, 20),
                },
                description="Random Forest for regression tasks.",
            ),
            AlgorithmMeta(
                name="gradient_boosting_regressor",
                factory=lambda **kw: GradientBoostingRegressor(**kw),
                tasks=["regression"],
                default_params={"n_estimators": 200, "random_state": 42},
                param_space={
                    "n_estimators": ("int", 50, 400),
                    "learning_rate": ("float_log", 0.01, 0.3),
                    "max_depth": ("int", 2, 8),
                },
                description="Gradient Boosting for regression tasks.",
            ),
            AlgorithmMeta(
                name="ridge",
                factory=lambda **kw: Ridge(**kw),
                tasks=["regression"],
                default_params={"alpha": 1.0},
                requires_scaling=True,
                param_space={"alpha": ("float_log", 1e-4, 100.0)},
                description="Ridge regression (L2 regularization). Fast and stable.",
            ),
            AlgorithmMeta(
                name="lasso",
                factory=lambda **kw: Lasso(**kw),
                tasks=["regression"],
                default_params={"alpha": 1.0},
                requires_scaling=True,
                param_space={"alpha": ("float_log", 1e-4, 10.0)},
                description="Lasso regression (L1 regularization). Automatic feature selection.",
            ),
            AlgorithmMeta(
                name="linear_regression",
                factory=lambda **kw: LinearRegression(n_jobs=-1, **kw),
                tasks=["regression"],
                default_params={},
                requires_scaling=True,
                description="Ordinary least squares. Simplest regression model.",
            ),
            AlgorithmMeta(
                name="extra_trees_regressor",
                factory=lambda **kw: ExtraTreesRegressor(n_jobs=-1, **kw),
                tasks=["regression"],
                default_params={"n_estimators": 200, "random_state": 42},
                param_space={"n_estimators": ("int", 50, 400)},
                description="Extra Trees for regression.",
            ),
        ]

        for meta in builtins:
            self.register(meta)

        # Try optional dependencies
        self._try_register_xgboost()
        self._try_register_lightgbm()

    def _try_register_xgboost(self) -> None:
        try:
            import xgboost as xgb
            self.register(AlgorithmMeta(
                name="xgboost",
                factory=lambda **kw: xgb.XGBClassifier(
                    eval_metric="logloss", verbosity=0, n_jobs=-1, **kw
                ),
                tasks=["classification"],
                default_params={"n_estimators": 200, "random_state": 42},
                gpu_supported=True,
                param_space={
                    "n_estimators": ("int", 50, 500),
                    "learning_rate": ("float_log", 0.01, 0.3),
                    "max_depth": ("int", 2, 10),
                    "subsample": ("float", 0.5, 1.0),
                    "colsample_bytree": ("float", 0.5, 1.0),
                },
                description="XGBoost gradient boosting. State-of-the-art on tabular data.",
            ))
            self.register(AlgorithmMeta(
                name="xgboost_regressor",
                factory=lambda **kw: xgb.XGBRegressor(verbosity=0, n_jobs=-1, **kw),
                tasks=["regression"],
                default_params={"n_estimators": 200, "random_state": 42},
                gpu_supported=True,
                param_space={
                    "n_estimators": ("int", 50, 500),
                    "learning_rate": ("float_log", 0.01, 0.3),
                    "max_depth": ("int", 2, 10),
                },
                description="XGBoost for regression.",
            ))
            logger.debug("XGBoost registered")
        except ImportError:
            pass

    def _try_register_lightgbm(self) -> None:
        try:
            import lightgbm as lgb
            self.register(AlgorithmMeta(
                name="lightgbm",
                factory=lambda **kw: lgb.LGBMClassifier(n_jobs=-1, verbose=-1, **kw),
                tasks=["classification"],
                default_params={"n_estimators": 200, "random_state": 42},
                gpu_supported=True,
                param_space={
                    "n_estimators": ("int", 50, 500),
                    "learning_rate": ("float_log", 0.01, 0.3),
                    "num_leaves": ("int", 20, 300),
                    "subsample": ("float", 0.5, 1.0),
                },
                description="LightGBM. Fastest gradient boosting on large datasets.",
            ))
            self.register(AlgorithmMeta(
                name="lightgbm_regressor",
                factory=lambda **kw: lgb.LGBMRegressor(n_jobs=-1, verbose=-1, **kw),
                tasks=["regression"],
                default_params={"n_estimators": 200, "random_state": 42},
                gpu_supported=True,
                param_space={
                    "n_estimators": ("int", 50, 500),
                    "learning_rate": ("float_log", 0.01, 0.3),
                    "num_leaves": ("int", 20, 300),
                },
                description="LightGBM for regression.",
            ))
            logger.debug("LightGBM registered")
        except ImportError:
            pass


# ── Registration Decorator ────────────────────────────────────────────────────

def register_algorithm(
    name: str,
    tasks: List[str],
    **meta_kwargs: Any,
):
    """
    Decorator for registering a custom algorithm.

    Usage:
        @register_algorithm("my_model", tasks=["classification"])
        def my_model_factory(**params):
            return MyModel(**params)
    """
    def decorator(factory: Callable) -> Callable:
        meta = AlgorithmMeta(name=name, factory=factory, tasks=tasks, **meta_kwargs)
        ModelRegistry.instance().register(meta)
        return factory
    return decorator
