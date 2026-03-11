"""
pycheron.train.trainer — The full ML training pipeline.

Orchestrates: load → validate → profile → preprocess → detect task
             → select algorithm → cross-validate → tune → evaluate → return model
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, train_test_split,
)

from pycheron.data.loader import DataLoader
from pycheron.data.validator import SchemaDetector, DataValidator
from pycheron.data.profiler import DataProfiler
from pycheron.evaluate.evaluator import Evaluator
from pycheron.optimize.tuner import Optimizer
from pycheron.preprocess.pipeline import Preprocessor
from pycheron.registry.model_registry import ModelRegistry
from pycheron.train.trained_model import TrainedModel
from pycheron.utils.logging import get_logger, StageLogger
from pycheron.utils.types import DataSource, TaskType

logger = get_logger(__name__)


class TaskDetector:
    """Detects ML task type from the target column."""

    CLASSIFICATION_THRESHOLD = 20  # unique values below this = classification

    def detect(self, y, task_override: Optional[str] = None) -> TaskType:
        import pandas as pd

        if task_override == "classification":
            n_unique = len(set(y))
            return (TaskType.BINARY_CLASSIFICATION if n_unique == 2
                    else TaskType.MULTICLASS_CLASSIFICATION)
        if task_override == "regression":
            return TaskType.REGRESSION

        # Auto-detect
        if hasattr(y, 'dtype'):
            if y.dtype == object or str(y.dtype).startswith('category'):
                n_unique = y.nunique() if hasattr(y, 'nunique') else len(set(y))
                return (TaskType.BINARY_CLASSIFICATION if n_unique == 2
                        else TaskType.MULTICLASS_CLASSIFICATION)

        n_unique = len(set(y))
        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION
        if n_unique <= self.CLASSIFICATION_THRESHOLD:
            return TaskType.MULTICLASS_CLASSIFICATION
        return TaskType.REGRESSION


class Trainer:
    """
    Orchestrates the full ML training pipeline.

    Pipeline stages:
    1. Load & validate data
    2. Detect or infer task type
    3. Build preprocessing pipeline
    4. Select algorithm (or use provided)
    5. Cross-validate baseline
    6. Optionally tune hyperparameters
    7. Refit on full train set
    8. Evaluate on held-out test set
    9. Return TrainedModel
    """

    def __init__(
        self,
        algorithm: Optional[str] = None,
        task: Optional[str] = None,
        test_size: float = 0.2,
        cv: int = 5,
        tune: bool = False,
        tune_trials: int = 30,
        time_budget: Optional[int] = None,
        metric: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1,
    ):
        self.algorithm_name = algorithm
        self.task_override = task
        self.test_size = test_size
        self.cv = cv
        self.tune = tune
        self.tune_trials = tune_trials
        self.time_budget = time_budget
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Sub-components
        self._loader = DataLoader()
        self._schema_detector = SchemaDetector()
        self._validator = DataValidator()
        self._profiler = DataProfiler()
        self._task_detector = TaskDetector()
        self._registry = ModelRegistry.instance()

    def fit(self, data: DataSource, target: str, **kwargs) -> TrainedModel:
        """Execute the full training pipeline."""
        t_start = time.time()

        # ── 1. Load ──────────────────────────────────────────────────────
        with StageLogger("Loading data", self.verbose):
            df = self._loader.load(data, target=target)

        # ── 2. Schema + Validate + Profile ───────────────────────────────
        with StageLogger("Schema detection & validation", self.verbose):
            schema = self._schema_detector.detect(df, target=target)
            validation = self._validator.validate(df, target=target, schema=schema)
            profile = self._profiler.profile(df, target=target, schema=schema)
            logger.info(str(profile))

        # ── 3. Detect task ────────────────────────────────────────────────
        task = self._task_detector.detect(df[target], task_override=self.task_override)
        logger.info(f"Task detected: {task.value}")

        # ── 4. Select algorithm ───────────────────────────────────────────
        if self.algorithm_name:
            algorithm_name = self.algorithm_name
        else:
            recommended = self._registry.recommend(profile, task, n=1)
            algorithm_name = recommended[0]
            logger.info(f"Auto-selected algorithm: {algorithm_name}")

        algo_meta = self._registry.get(algorithm_name)

        # ── 5. Preprocess ─────────────────────────────────────────────────
        with StageLogger("Preprocessing", self.verbose):
            preprocessor = Preprocessor(algorithm=algorithm_name, task=task)
            X_all, y_all = preprocessor.fit_transform(df, target, schema, profile, validation)

        # ── 6. Train/test split ───────────────────────────────────────────
        if task.is_classification:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all,
                test_size=self.test_size,
                stratify=y_all,
                random_state=self.random_state,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all,
                test_size=self.test_size,
                random_state=self.random_state,
            )

        # ── 7. Cross-validation baseline ──────────────────────────────────
        with StageLogger(f"Cross-validation ({self.cv}-fold)", self.verbose):
            evaluator = Evaluator(task=task, metric=self.metric, algorithm=algorithm_name)
            scoring = evaluator.cv_scoring_metric()

            splitter = (
                StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                if task.is_classification else
                KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            )

            base_estimator = algo_meta.build(random_state=self.random_state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    base_estimator, X_train, y_train,
                    cv=splitter,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                )

            logger.info(
                f"CV {scoring}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
            )

        # ── 8. Hyperparameter tuning (optional) ───────────────────────────
        best_params = algo_meta.default_params.copy()
        best_params["random_state"] = self.random_state

        if self.tune:
            with StageLogger(f"Hyperparameter tuning ({self.tune_trials} trials)", self.verbose):
                optimizer = Optimizer(
                    algorithm_meta=algo_meta,
                    task=task,
                    n_trials=self.tune_trials,
                    cv=self.cv,
                    metric=self.metric,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    time_budget=self.time_budget,
                    verbose=self.verbose,
                )
                best_params = optimizer.optimize(X_train, y_train)

        # ── 9. Final fit ──────────────────────────────────────────────────
        with StageLogger("Final model fit", self.verbose):
            final_estimator = algo_meta.build(**best_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_estimator.fit(X_train, y_train)

        # ── 10. Test set evaluation ───────────────────────────────────────
        with StageLogger("Evaluation", self.verbose):
            y_pred = final_estimator.predict(X_test)
            y_proba = None
            if task.is_classification and hasattr(final_estimator, "predict_proba"):
                try:
                    y_proba = final_estimator.predict_proba(X_test)
                except Exception:
                    pass

            fi = self._get_feature_importances(final_estimator, preprocessor.feature_names)

            report = evaluator.evaluate(
                y_test, y_pred, y_proba=y_proba,
                cv_scores=cv_scores.tolist(),
                feature_importances=fi,
            )

        elapsed = time.time() - t_start
        logger.info(f"Training complete in {elapsed:.1f}s")

        if self.verbose >= 1:
            report.display()

        return TrainedModel(
            estimator=final_estimator,
            preprocessor=preprocessor,
            task=task,
            target=target,
            algorithm=algorithm_name,
            evaluation=report,
            feature_names=preprocessor.feature_names,
            meta={
                "training_time": elapsed,
                "n_rows": len(df),
                "data_hash": profile.data_hash,
                "random_state": self.random_state,
                "best_params": best_params,
            },
        )

    @staticmethod
    def _get_feature_importances(estimator, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """Extract feature importances if the estimator supports them."""
        if hasattr(estimator, "feature_importances_"):
            fi = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = np.abs(estimator.coef_)
            fi = coef.flatten() if coef.ndim > 1 else coef
        else:
            return None

        if len(feature_names) != len(fi):
            feature_names = [f"f{i}" for i in range(len(fi))]

        return dict(sorted(
            zip(feature_names, fi.tolist()),
            key=lambda x: x[1],
            reverse=True,
        ))
