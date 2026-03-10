"""
pymlkit.explain — SHAP-based model explainability.

Automatically chooses the right SHAP explainer type:
- Tree models (RF, GBT, XGB, LGBM) → TreeExplainer (fast, exact)
- Linear models → LinearExplainer
- Everything else → KernelExplainer (slower, model-agnostic)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from pymlkit.data.loader import DataLoader
from pymlkit.utils.logging import get_logger
from pymlkit.utils.types import DataSource

logger = get_logger(__name__)

_TREE_ESTIMATOR_NAMES = {
    "RandomForestClassifier", "RandomForestRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "DecisionTreeClassifier", "DecisionTreeRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
}

_LINEAR_ESTIMATOR_NAMES = {
    "LogisticRegression", "LinearRegression",
    "Ridge", "Lasso", "ElasticNet",
}


class Explainer:
    """
    Wraps SHAP to compute feature attributions for any model.

    Usage:
        explainer = Explainer(trained_model)
        explanation = explainer.explain(data)
        explanation.plot()          # bar chart of mean |SHAP|
        explanation.summary()       # table of top features
        explanation.waterfall(0)    # waterfall for sample 0
    """

    def __init__(self, trained_model: Any, n_samples: int = 100):
        self.model = trained_model
        self.n_samples = n_samples
        self._loader = DataLoader()

    def explain(self, data: Optional[DataSource] = None) -> "Explanation":
        """
        Compute SHAP values.

        Parameters
        ----------
        data : optional; uses a sample of training background if None.
        """
        try:
            import shap
        except ImportError:
            raise ImportError(
                "SHAP is required for explanations. Install it: pip install shap"
            )

        estimator = self.model.estimator
        preprocessor = self.model.preprocessor
        feature_names = self.model.feature_names

        # Get or load data to explain
        if data is not None:
            df = self._loader.load(data)
            X = preprocessor.transform(df)
        elif hasattr(preprocessor, "_transformer") and preprocessor._transformer is not None:
            # Use training data — we don't store it, so generate from transformer
            # Fall back to explaining with zeros (last resort)
            X = np.zeros((min(self.n_samples, 50), len(feature_names)))
            logger.warning("No data provided; explaining with zero baseline (low quality)")
        else:
            X = np.zeros((10, len(feature_names)))

        X = X[:self.n_samples]

        est_name = type(estimator).__name__
        shap_values = None
        background = shap.kmeans(X, min(10, len(X)))

        try:
            if est_name in _TREE_ESTIMATOR_NAMES:
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X)
            elif est_name in _LINEAR_ESTIMATOR_NAMES:
                explainer = shap.LinearExplainer(estimator, X)
                shap_values = explainer.shap_values(X)
            else:
                explainer = shap.KernelExplainer(
                    estimator.predict_proba if hasattr(estimator, "predict_proba")
                    else estimator.predict,
                    background,
                )
                shap_values = explainer.shap_values(X[:20])  # KernelSHAP is slow
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}. Falling back to feature importances.")
            shap_values = None

        # Compute mean |SHAP| importance
        if shap_values is not None:
            if isinstance(shap_values, list):
                # Multi-class: average across classes
                sv = np.array(shap_values)
                importance = np.mean(np.abs(sv), axis=(0, 1))
            else:
                importance = np.mean(np.abs(shap_values), axis=0)
        elif self.model.evaluation and self.model.evaluation.feature_importances:
            # Fall back to model's native feature importances
            fi = self.model.evaluation.feature_importances
            importance = np.array([fi.get(f, 0.0) for f in feature_names])
            shap_values = None
        else:
            importance = np.ones(len(feature_names)) / len(feature_names)
            shap_values = None

        return Explanation(
            shap_values=shap_values,
            feature_names=feature_names,
            importance=importance,
            X=X,
            algorithm=self.model.algorithm,
        )


class Explanation:
    """
    Holds SHAP values and feature importances for a model.

    Methods
    -------
    plot()        — bar chart of mean absolute SHAP values
    summary()     — DataFrame of top features ranked by importance
    waterfall(i)  — SHAP waterfall plot for sample i
    """

    def __init__(
        self,
        shap_values,
        feature_names: List[str],
        importance: np.ndarray,
        X: np.ndarray,
        algorithm: str,
    ):
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.importance = importance
        self.X = X
        self.algorithm = algorithm

    def summary(self, top_n: int = 20) -> pd.DataFrame:
        """Return a DataFrame of features ranked by mean |SHAP| importance."""
        n = min(top_n, len(self.feature_names))
        idx = np.argsort(self.importance)[::-1][:n]
        df = pd.DataFrame({
            "feature": [self.feature_names[i] for i in idx],
            "importance": self.importance[idx],
        })
        df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
        df.index += 1
        return df

    def plot(self, top_n: int = 20) -> None:
        """Display a bar chart of the top features by importance."""
        summary = self.summary(top_n)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
            y_pos = range(len(summary))
            ax.barh(y_pos, summary["importance"], color="#2563EB")
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(summary["feature"])
            ax.invert_yaxis()
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title(f"Feature Importance — {self.algorithm}")
            plt.tight_layout()
            plt.show()
        except ImportError:
            # Fallback: text table
            try:
                from rich.console import Console
                from rich.table import Table
                console = Console()
                t = Table(title=f"Feature Importance — {self.algorithm}")
                t.add_column("Feature", style="cyan")
                t.add_column("Importance", justify="right")
                t.add_column("%", justify="right")
                for _, row in summary.iterrows():
                    t.add_row(row["feature"], f"{row['importance']:.4f}", f"{row['importance_pct']:.1f}%")
                console.print(t)
            except ImportError:
                print(summary.to_string())

    def waterfall(self, sample_idx: int = 0) -> None:
        """Display SHAP waterfall plot for a single sample."""
        if self.shap_values is None:
            logger.warning("SHAP values not available for waterfall plot.")
            return
        try:
            import shap
            import matplotlib.pyplot as plt
            sv = self.shap_values
            if isinstance(sv, list):
                sv = sv[1]  # class 1 for binary
            shap.plots.waterfall(
                shap.Explanation(
                    values=sv[sample_idx],
                    feature_names=self.feature_names,
                )
            )
        except Exception as e:
            logger.warning(f"Waterfall plot failed: {e}")

    def __repr__(self) -> str:
        return (
            f"Explanation(algorithm={self.algorithm!r}, "
            f"features={len(self.feature_names)}, "
            f"top={self.summary(3)['feature'].tolist()})"
        )
