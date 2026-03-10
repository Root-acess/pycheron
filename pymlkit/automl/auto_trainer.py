"""
pymlkit.automl.auto_trainer — AutoML tournament engine.

Runs a 3-stage tournament:
  Stage 1: Quick baseline for all candidates on a data sample
  Stage 2: Keep top-N, full CV evaluation
  Stage 3: Bayesian tune the winner
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, train_test_split

from pymlkit.data.loader import DataLoader
from pymlkit.data.validator import SchemaDetector, DataValidator
from pymlkit.data.profiler import DataProfiler
from pymlkit.evaluate.evaluator import Evaluator
from pymlkit.optimize.tuner import Optimizer
from pymlkit.preprocess.pipeline import Preprocessor
from pymlkit.registry.model_registry import ModelRegistry
from pymlkit.train.trained_model import TrainedModel
from pymlkit.train.trainer import TaskDetector
from pymlkit.utils.logging import get_logger, StageLogger
from pymlkit.utils.types import DataSource, TaskType

logger = get_logger(__name__)


class AutoTrainer:
    """
    AutoML engine that selects the best algorithm via tournament evaluation.

    Algorithm
    ---------
    1. Load + preprocess data
    2. Get top N algorithm candidates from registry
    3. Stage 1: Quick eval on 30% sample → rank candidates
    4. Stage 2: Full CV on top 3 → rank again
    5. Stage 3: Bayesian tune winner
    6. Refit winner on full training set, evaluate on test set
    """

    def __init__(
        self,
        time_budget: int = 120,
        n_candidates: int = 6,
        tune_trials: int = 30,
        metric: Optional[str] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1,
        cv: int = 5,
        test_size: float = 0.2,
    ):
        self.time_budget = time_budget
        self.n_candidates = n_candidates
        self.tune_trials = tune_trials
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv = cv
        self.test_size = test_size

        self._loader = DataLoader()
        self._schema_detector = SchemaDetector()
        self._validator = DataValidator()
        self._profiler = DataProfiler()
        self._task_detector = TaskDetector()
        self._registry = ModelRegistry.instance()

    def fit(self, data: DataSource, target: Optional[str] = None) -> TrainedModel:
        t_start = time.time()
        budget_per_stage = self.time_budget / 3

        # ── 1. Load + detect ──────────────────────────────────────────────
        with StageLogger("AutoML: Loading & profiling", self.verbose):
            df = self._loader.load(data)

            if target is None:
                target = df.columns[-1]
                logger.info(f"Auto-detected target column: '{target}'")

            schema = self._schema_detector.detect(df, target=target)
            validation = self._validator.validate(df, target=target, schema=schema)
            profile = self._profiler.profile(df, target=target, schema=schema)
            task = self._task_detector.detect(df[target])
            logger.info(f"Task: {task.value} | {profile}")

        # ── 2. Preprocess ─────────────────────────────────────────────────
        # We use the first candidate's name for preprocessing; most decisions
        # are algorithm-agnostic (only scaling differs, and we override below)
        with StageLogger("AutoML: Preprocessing", self.verbose):
            preprocessor = Preprocessor(algorithm="random_forest", task=task)
            X_all, y_all = preprocessor.fit_transform(df, target, schema, profile, validation)

        # Train/test split
        if task.is_classification:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=self.test_size,
                stratify=y_all, random_state=self.random_state,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=self.test_size,
                random_state=self.random_state,
            )

        evaluator = Evaluator(task=task, metric=self.metric)

        # ── 3. Candidate recommendation ───────────────────────────────────
        candidates = self._registry.recommend(profile, task, n=self.n_candidates)
        logger.info(f"AutoML candidates: {candidates}")

        # ── 4. Stage 1: Quick sample eval ────────────────────────────────
        with StageLogger(f"AutoML Stage 1: Quick eval ({len(candidates)} models)", self.verbose):
            sample_size = min(5000, len(X_train))
            idx = np.random.RandomState(self.random_state).choice(len(X_train), sample_size, replace=False)
            X_sample, y_sample = X_train[idx], y_train[idx]

            stage1_scores: List[tuple] = []
            t1 = time.time()
            for name in candidates:
                if time.time() - t1 > budget_per_stage:
                    logger.warning("Stage 1 time budget exceeded — truncating candidates")
                    break
                try:
                    meta = self._registry.get(name)
                    est = meta.build(random_state=self.random_state)
                    scoring = evaluator.cv_scoring_metric()
                    splitter = (
                        StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                        if task.is_classification else
                        KFold(n_splits=3, shuffle=True, random_state=self.random_state)
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        s = cross_val_score(est, X_sample, y_sample, cv=splitter,
                                            scoring=scoring, n_jobs=self.n_jobs)
                    score = float(np.mean(s))
                    stage1_scores.append((score, name))
                    logger.info(f"  {name}: {score:.4f}")
                except Exception as e:
                    logger.warning(f"  {name} failed in Stage 1: {e}")

            stage1_scores.sort(reverse=True)
            top3 = [name for _, name in stage1_scores[:3]]

        # ── 5. Stage 2: Full CV on top 3 ─────────────────────────────────
        with StageLogger(f"AutoML Stage 2: Full CV (top {len(top3)} models)", self.verbose):
            stage2_scores: List[tuple] = []
            t2 = time.time()
            for name in top3:
                if time.time() - t2 > budget_per_stage:
                    break
                try:
                    meta = self._registry.get(name)
                    est = meta.build(random_state=self.random_state)
                    splitter = (
                        StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                        if task.is_classification else
                        KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        s = cross_val_score(est, X_train, y_train, cv=splitter,
                                            scoring=evaluator.cv_scoring_metric(),
                                            n_jobs=self.n_jobs)
                    score = float(np.mean(s))
                    stage2_scores.append((score, name, s.tolist()))
                    logger.info(f"  {name}: {score:.4f} ± {np.std(s):.4f}")
                except Exception as e:
                    logger.warning(f"  {name} failed in Stage 2: {e}")

            if not stage2_scores:
                raise RuntimeError("All AutoML candidates failed. Check your data.")
            stage2_scores.sort(reverse=True)
            winner_score, winner_name, winner_cv = stage2_scores[0]
            logger.info(f"Stage 2 winner: {winner_name} ({winner_score:.4f})")

        # ── 6. Stage 3: Tune winner ───────────────────────────────────────
        winner_meta = self._registry.get(winner_name)
        best_params = winner_meta.default_params.copy()
        best_params["random_state"] = self.random_state

        remaining_time = int(self.time_budget - (time.time() - t_start))
        if remaining_time > 10 and winner_meta.param_space:
            with StageLogger(f"AutoML Stage 3: Tuning {winner_name}", self.verbose):
                optimizer = Optimizer(
                    algorithm_meta=winner_meta,
                    task=task,
                    n_trials=self.tune_trials,
                    cv=self.cv,
                    metric=self.metric,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                    time_budget=remaining_time,
                    verbose=self.verbose,
                )
                best_params = optimizer.optimize(X_train, y_train)

        # ── 7. Final fit + evaluate ───────────────────────────────────────
        with StageLogger("AutoML: Final fit & evaluation", self.verbose):
            final_est = winner_meta.build(**best_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_est.fit(X_train, y_train)

            y_pred = final_est.predict(X_test)
            y_proba = None
            if task.is_classification and hasattr(final_est, "predict_proba"):
                try:
                    y_proba = final_est.predict_proba(X_test)
                except Exception:
                    pass

            fi = None
            if hasattr(final_est, "feature_importances_"):
                fi = dict(zip(preprocessor.feature_names, final_est.feature_importances_.tolist()))
            elif hasattr(final_est, "coef_"):
                coef = np.abs(final_est.coef_).flatten()
                fi = dict(zip(preprocessor.feature_names, coef.tolist()))

            final_evaluator = Evaluator(task=task, metric=self.metric, algorithm=winner_name)
            report = final_evaluator.evaluate(
                y_test, y_pred, y_proba=y_proba,
                cv_scores=winner_cv,
                feature_importances=fi,
            )

        elapsed = time.time() - t_start
        logger.info(f"AutoML complete in {elapsed:.1f}s")

        if self.verbose >= 1:
            report.display()

        # Build leaderboard data
        leaderboard = []
        for score, name, cv_s in stage2_scores:
            leaderboard.append({
                "rank": len(leaderboard) + 1,
                "algorithm": name,
                "score": score,
                "cv_std": float(np.std(cv_s)),
                "metric": evaluator.metric,
            })

        return TrainedModel(
            estimator=final_est,
            preprocessor=preprocessor,
            task=task,
            target=target,
            algorithm=winner_name,
            evaluation=report,
            feature_names=preprocessor.feature_names,
            leaderboard_data=leaderboard,
            meta={
                "training_time": elapsed,
                "n_rows": len(df),
                "data_hash": profile.data_hash,
                "random_state": self.random_state,
                "best_params": best_params,
                "automl": True,
                "candidates_evaluated": [name for _, name in stage1_scores],
            },
        )
