"""
pymlkit.train.trained_model — The TrainedModel object returned to users.

Wraps the fitted estimator + preprocessor + metadata.
Provides predict(), evaluate(), explain(), save(), leaderboard().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from pymlkit.data.loader import DataLoader
from pymlkit.evaluate.evaluator import EvaluationReport
from pymlkit.utils.logging import get_logger
from pymlkit.utils.types import DataSource, TaskType

logger = get_logger(__name__)


class TrainedModel:
    """
    The primary object returned by pymlkit.train() and pymlkit.auto_train().

    Methods
    -------
    predict(data)       → labels (array or DataFrame)
    predict_proba(data) → class probabilities
    evaluate()          → display evaluation report
    explain(data)       → SHAP explanations
    save(path)          → persist to disk
    leaderboard()       → show AutoML tournament results (auto_train only)
    """

    def __init__(
        self,
        estimator: Any,
        preprocessor: Any,
        task: TaskType,
        target: str,
        algorithm: str,
        evaluation: Optional[EvaluationReport] = None,
        feature_names: Optional[List[str]] = None,
        leaderboard_data: Optional[List[Dict]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.task = task
        self.target = target
        self.algorithm = algorithm
        self.evaluation = evaluation
        self.feature_names = feature_names or []
        self._leaderboard_data = leaderboard_data or []
        self.meta = meta or {}
        self._loader = DataLoader()

    # ── Prediction ────────────────────────────────────────────────────────

    def predict(self, data: DataSource) -> np.ndarray:
        """
        Predict labels for new data.

        Parameters
        ----------
        data : str | Path | pd.DataFrame
            New data to predict on. Must have same columns as training data.

        Returns
        -------
        np.ndarray of predicted labels.
        """
        X = self._prepare_input(data)
        y_pred = self.estimator.predict(X)
        # Decode labels if they were encoded
        y_decoded = self.preprocessor.decode_target(y_pred)
        return y_decoded

    def predict_proba(self, data: DataSource) -> np.ndarray:
        """
        Return class probability estimates (classification only).

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes).
        """
        if not self.task.is_classification:
            raise ValueError("predict_proba() is only available for classification tasks.")
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(f"Algorithm '{self.algorithm}' does not support predict_proba.")
        X = self._prepare_input(data)
        return self.estimator.predict_proba(X)

    def _prepare_input(self, data: DataSource) -> np.ndarray:
        """Load and preprocess input data."""
        df = self._loader.load(data)
        return self.preprocessor.transform(df)

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate(self) -> EvaluationReport:
        """Display the evaluation report from training."""
        if self.evaluation is None:
            raise RuntimeError("No evaluation report available. Was the model trained?")
        self.evaluation.display()
        return self.evaluation

    # ── Explainability ────────────────────────────────────────────────────

    def explain(
        self,
        data: Optional[DataSource] = None,
        n_samples: int = 100,
    ) -> "Explanation":
        """
        Generate SHAP-based feature importance explanations.

        Parameters
        ----------
        data : optional data to explain; uses training sample if None.
        n_samples : number of background samples for KernelSHAP.
        """
        from pymlkit.explain.explainer import Explainer
        explainer = Explainer(self, n_samples=n_samples)
        return explainer.explain(data)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save the model to disk.

        Parameters
        ----------
        path : directory path (created if doesn't exist).

        Returns
        -------
        Path to the saved model directory.
        """
        from pymlkit.persistence.model_store import ModelStore
        return ModelStore.save(self, path)

    # ── AutoML leaderboard ────────────────────────────────────────────────

    def leaderboard(self) -> Optional[pd.DataFrame]:
        """
        Show the AutoML tournament leaderboard (auto_train only).

        Returns a DataFrame sorted by primary metric (best first).
        """
        if not self._leaderboard_data:
            logger.info("No leaderboard available. Use auto_train() for tournament results.")
            return None
        df = pd.DataFrame(self._leaderboard_data)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df.index += 1
        try:
            from rich.console import Console
            from rich.table import Table
            console = Console()
            t = Table(title="AutoML Leaderboard")
            for col in df.columns:
                t.add_column(col, style="cyan" if col == "algorithm" else "white")
            for _, row in df.iterrows():
                vals = []
                for col in df.columns:
                    v = row[col]
                    vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
                t.add_row(*vals)
            console.print(t)
        except ImportError:
            print(df.to_string())
        return df

    def __repr__(self) -> str:
        score_str = ""
        if self.evaluation:
            score_str = f", {self.evaluation.primary_metric}={self.evaluation.primary_score:.4f}"
        return f"TrainedModel(algorithm={self.algorithm!r}, task={self.task.value!r}{score_str})"
