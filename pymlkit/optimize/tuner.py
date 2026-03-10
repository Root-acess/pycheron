"""
pymlkit.optimize — Hyperparameter optimization via Optuna.

Supports:
- RandomSearch (fast baseline)
- BayesianOptimization (default for auto_train)
- GridSearch (sklearn compat, small spaces)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import cross_val_score

from pymlkit.evaluate.evaluator import Evaluator
from pymlkit.registry.model_registry import AlgorithmMeta
from pymlkit.utils.logging import get_logger
from pymlkit.utils.types import TaskType

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False
logger = get_logger(__name__)


class Optimizer:
    """
    Bayesian hyperparameter optimizer using Optuna.

    Builds an Optuna trial from AlgorithmMeta.param_space and
    searches for the best hyperparameters using cross-validation.
    """

    def __init__(
        self,
        algorithm_meta: AlgorithmMeta,
        task: TaskType,
        n_trials: int = 30,
        cv: int = 5,
        metric: Optional[str] = None,
        n_jobs: int = -1,
        random_state: int = 42,
        time_budget: Optional[int] = None,
        verbose: int = 1,
    ):
        self.meta = algorithm_meta
        self.task = task
        self.n_trials = n_trials
        self.cv = cv
        self.evaluator = Evaluator(task=task, metric=metric)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.time_budget = time_budget
        self.verbose = verbose
        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = 0.0

    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Run hyperparameter search.

        Returns the best parameter dict found.
        """
        if not self.meta.param_space:
            logger.info(f"No param space defined for '{self.meta.name}' — skipping tuning")
            return self.meta.default_params.copy()

        if not _OPTUNA:
            logger.warning("optuna not installed — skipping hyperparameter tuning. Install: pip install optuna")
            return {**self.meta.default_params, "random_state": self.random_state}

        scoring = self.evaluator.cv_scoring_metric()
        direction = "maximize" if not scoring.startswith("neg_") else "maximize"

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial)
            estimator = self.meta.build(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(
                    estimator, X, y,
                    cv=self.cv,
                    scoring=scoring,
                    n_jobs=self.n_jobs,
                )
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction=direction, sampler=sampler)

        kwargs: Dict[str, Any] = {"n_trials": self.n_trials, "show_progress_bar": self.verbose >= 1}
        if self.time_budget:
            kwargs["timeout"] = self.time_budget

        study.optimize(objective, **kwargs)

        self.best_params_ = {**self.meta.default_params, **study.best_params}
        self.best_score_ = study.best_value

        logger.info(
            f"Best params for '{self.meta.name}': {study.best_params} "
            f"(score={self.best_score_:.4f})"
        )
        return self.best_params_

    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Convert AlgorithmMeta.param_space entries into Optuna suggestions."""
        params: Dict[str, Any] = {}
        for param_name, spec in self.meta.param_space.items():
            ptype = spec[0]
            if ptype == "int":
                params[param_name] = trial.suggest_int(param_name, spec[1], spec[2])
            elif ptype == "int_none":
                # Allows None as well
                val = trial.suggest_int(param_name + "_raw", 0, spec[2] - spec[1] + 1)
                params[param_name] = None if val == 0 else spec[1] + val - 1
            elif ptype == "float":
                params[param_name] = trial.suggest_float(param_name, spec[1], spec[2])
            elif ptype == "float_log":
                params[param_name] = trial.suggest_float(param_name, spec[1], spec[2], log=True)
            elif ptype == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, spec[1])
        return params
