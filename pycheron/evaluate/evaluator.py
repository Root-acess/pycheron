"""
pycheron.evaluate — Automatic metric selection and evaluation reporting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error,
)

from pycheron.utils.logging import get_logger
from pycheron.utils.types import TaskType

logger = get_logger(__name__)


class EvaluationReport:
    """
    Structured result from an evaluation run.

    Holds all metrics, and provides display and comparison methods.
    """

    def __init__(
        self,
        task: TaskType,
        metrics: Dict[str, float],
        primary_metric: str,
        confusion: Optional[np.ndarray] = None,
        class_report: Optional[str] = None,
        feature_importances: Optional[Dict[str, float]] = None,
        cv_scores: Optional[List[float]] = None,
        algorithm: str = "",
    ):
        self.task = task
        self.metrics = metrics
        self.primary_metric = primary_metric
        self.confusion = confusion
        self.class_report = class_report
        self.feature_importances = feature_importances
        self.cv_scores = cv_scores
        self.algorithm = algorithm

    @property
    def primary_score(self) -> float:
        return self.metrics.get(self.primary_metric, 0.0)

    def display(self) -> None:
        """Print a formatted evaluation summary."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel

            console = Console()
            table = Table(title=f"Evaluation Report — {self.algorithm}", show_header=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Score", style="bold green", justify="right")

            for k, v in self.metrics.items():
                marker = " ★" if k == self.primary_metric else ""
                table.add_row(k + marker, f"{v:.4f}")

            if self.cv_scores:
                mean_cv = np.mean(self.cv_scores)
                std_cv = np.std(self.cv_scores)
                table.add_row(
                    f"CV {self.primary_metric} (mean±std)",
                    f"{mean_cv:.4f} ± {std_cv:.4f}"
                )

            console.print(table)

            if self.class_report:
                console.print(Panel(self.class_report, title="Classification Report"))

        except ImportError:
            print(f"\n=== Evaluation Report ({self.algorithm}) ===")
            for k, v in self.metrics.items():
                marker = " [PRIMARY]" if k == self.primary_metric else ""
                print(f"  {k}{marker}: {v:.4f}")
            if self.cv_scores:
                print(f"  CV {self.primary_metric}: {np.mean(self.cv_scores):.4f} ± {np.std(self.cv_scores):.4f}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "task": self.task.value,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "metrics": self.metrics,
            "cv_scores": self.cv_scores,
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationReport({self.algorithm}, "
            f"{self.primary_metric}={self.primary_score:.4f})"
        )


class Evaluator:
    """
    Computes evaluation metrics appropriate for the task type.

    Automatically selects:
    - Binary classification: ROC-AUC, F1, Accuracy, Precision, Recall
    - Multi-class: Macro-F1, Accuracy, per-class report
    - Regression: RMSE, MAE, R², MAPE
    """

    def __init__(
        self,
        task: TaskType,
        metric: Optional[str] = None,
        algorithm: str = "",
    ):
        self.task = task
        self.metric = metric or self._default_metric(task)
        self.algorithm = algorithm

    @staticmethod
    def _default_metric(task: TaskType) -> str:
        if task == TaskType.BINARY_CLASSIFICATION:
            return "roc_auc"
        elif task == TaskType.MULTICLASS_CLASSIFICATION:
            return "f1_macro"
        else:
            return "rmse"

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        cv_scores: Optional[List[float]] = None,
        feature_importances: Optional[Dict[str, float]] = None,
    ) -> EvaluationReport:

        if self.task.is_classification:
            metrics = self._classification_metrics(y_true, y_pred, y_proba)
            cm = confusion_matrix(y_true, y_pred)
            cr = classification_report(y_true, y_pred)
        else:
            metrics = self._regression_metrics(y_true, y_pred)
            cm = None
            cr = None

        return EvaluationReport(
            task=self.task,
            metrics=metrics,
            primary_metric=self.metric,
            confusion=cm,
            class_report=cr,
            feature_importances=feature_importances,
            cv_scores=cv_scores,
            algorithm=self.algorithm,
        )

    def _classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
    ) -> Dict[str, float]:
        avg = "binary" if self.task == TaskType.BINARY_CLASSIFICATION else "macro"
        metrics: Dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        }
        if y_proba is not None:
            try:
                if self.task == TaskType.BINARY_CLASSIFICATION:
                    proba_1d = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    metrics["roc_auc"] = roc_auc_score(y_true, proba_1d)
                else:
                    metrics["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="macro"
                    )
            except Exception:
                pass
        return metrics

    @staticmethod
    def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
        }
        try:
            metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
        except Exception:
            pass
        return metrics

    def cv_scoring_metric(self) -> str:
        """Return the sklearn-compatible scoring string for cross-validation."""
        mapping = {
            "roc_auc": "roc_auc",
            "f1_macro": "f1_macro",
            "accuracy": "accuracy",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
        return mapping.get(self.metric, "accuracy")
