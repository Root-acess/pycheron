"""
pycheron._api — Public-facing API functions.

These are the three functions most users will ever need.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from pycheron.utils.logging import get_logger

logger = get_logger(__name__)


def train(
    data: Union[str, Path, pd.DataFrame],
    target: str,
    *,
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
    save_path: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> "TrainedModel":
    """
    Train an ML model with minimal configuration.

    Parameters
    ----------
    data : str | Path | pd.DataFrame
        Path to a CSV/Parquet file, or a DataFrame.
    target : str
        Name of the target (label) column.
    algorithm : str, optional
        Algorithm name from the registry. If None, auto-selects best fit.
        Options: 'random_forest', 'logistic_regression', 'gradient_boosting',
                 'ridge', 'lasso', 'knn', 'svm', 'decision_tree',
                 'xgboost', 'lightgbm'  (if installed)
    task : str, optional
        'classification' or 'regression'. Auto-detected if None.
    test_size : float
        Fraction of data held out for final evaluation. Default 0.2.
    cv : int
        Number of cross-validation folds. Default 5.
    tune : bool
        Run hyperparameter tuning. Default False.
    tune_trials : int
        Number of Optuna trials if tune=True. Default 30.
    time_budget : int, optional
        Maximum seconds for the entire training run.
    metric : str, optional
        Primary evaluation metric. Auto-selected by task if None.
    random_state : int
        Global random seed for reproducibility. Default 42.
    n_jobs : int
        Parallel jobs (-1 = all CPUs). Default -1.
    verbose : int
        0 = silent, 1 = progress bars, 2 = debug. Default 1.
    save_path : str | Path, optional
        If given, saves the trained model here after training.

    Returns
    -------
    TrainedModel
        Fitted model with .predict(), .evaluate(), .explain(), .save() methods.

    Examples
    --------
    >>> import pycheron
    >>> model = pycheron.train("titanic.csv", target="Survived")
    >>> model.evaluate()
    >>> predictions = model.predict("test.csv")
    """
    from pycheron.train.trainer import Trainer

    trainer = Trainer(
        algorithm=algorithm,
        task=task,
        test_size=test_size,
        cv=cv,
        tune=tune,
        tune_trials=tune_trials,
        time_budget=time_budget,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    model = trainer.fit(data, target, **kwargs)

    if save_path:
        model.save(save_path)

    return model


def auto_train(
    data: Union[str, Path, pd.DataFrame],
    target: Optional[str] = None,
    *,
    time_budget: int = 120,
    n_candidates: int = 6,
    tune_trials: int = 30,
    metric: Optional[str] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 1,
    save_path: Optional[Union[str, Path]] = None,
) -> "TrainedModel":
    """
    Fully automatic ML: loads data, selects best algorithm, tunes, evaluates.

    Parameters
    ----------
    data : str | Path | pd.DataFrame
        Path to data file or DataFrame.
    target : str, optional
        Target column name. If None, infers last column or raises.
    time_budget : int
        Total seconds for the entire AutoML run. Default 120.
    n_candidates : int
        Number of algorithms to try in the tournament. Default 6.
    tune_trials : int
        Optuna trials for the winning algorithm. Default 30.
    metric : str, optional
        Primary metric. Auto-selected if None.
    random_state : int
        Reproducibility seed. Default 42.
    n_jobs : int
        Parallel jobs. Default -1.
    verbose : int
        Verbosity level. Default 1.
    save_path : str | Path, optional
        Auto-save the model here after training.

    Returns
    -------
    TrainedModel
        Best fitted model from the tournament.

    Examples
    --------
    >>> import pycheron
    >>> model = pycheron.auto_train("data.csv", time_budget=60)
    >>> print(model.leaderboard())
    """
    from pycheron.automl.auto_trainer import AutoTrainer

    trainer = AutoTrainer(
        time_budget=time_budget,
        n_candidates=n_candidates,
        tune_trials=tune_trials,
        metric=metric,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    model = trainer.fit(data, target)

    if save_path:
        model.save(save_path)

    return model


def load_model(path: Union[str, Path]) -> "TrainedModel":
    """
    Load a previously saved pycheron model.

    Parameters
    ----------
    path : str | Path
        Directory or .pkl file path written by model.save().

    Returns
    -------
    TrainedModel
        Ready-to-predict model.

    Examples
    --------
    >>> model = pycheron.load_model("./my_model")
    >>> model.predict(new_data)
    """
    from pycheron.persistence.model_store import ModelStore

    return ModelStore.load(path)


def explain(
    model: "TrainedModel",
    data: Union[str, Path, pd.DataFrame, None] = None,
    *,
    n_samples: int = 100,
) -> "Explanation":
    """
    Generate SHAP-based explanations for a trained model.

    Parameters
    ----------
    model : TrainedModel
        A fitted pycheron model.
    data : str | Path | pd.DataFrame, optional
        Data to explain. Uses training data sample if None.
    n_samples : int
        Number of samples for KernelSHAP background. Default 100.

    Returns
    -------
    Explanation
        Object with .plot(), .summary(), .waterfall(i) methods.
    """
    from pycheron.explain.explainer import Explainer

    explainer = Explainer(model, n_samples=n_samples)
    return explainer.explain(data)
