"""
pycheron.evaluate — Metrics, evaluation, and model comparison.

Usage:
    import pycheron as pycrn
    pycrn.evaluate.compare([model1, model2], X_test, y_test)
"""

from __future__ import annotations
from pycheron.evaluate.evaluator import Evaluator, EvaluationReport


def compare(models: list, X_test, y_test):
    """
    Compare multiple trained models side by side on the same test set.

    Examples
    --------
    >>> pycrn.evaluate.compare([model1, model2], X_test, y_test)
    """
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score

    rows = []
    for model in models:
        y_pred = model.estimator.predict(X_test)
        row = {"algorithm": model.algorithm, "task": model.task.value}
        if model.task.is_classification:
            row["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
            row["f1_macro"] = round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4)
            try:
                if hasattr(model.estimator, "predict_proba"):
                    proba = model.estimator.predict_proba(X_test)
                    row["roc_auc"] = round(roc_auc_score(y_test, proba[:, 1] if proba.shape[1] == 2 else proba, multi_class="ovr"), 4)
            except Exception:
                row["roc_auc"] = None
        else:
            row["r2"] = round(r2_score(y_test, y_pred), 4)
            row["rmse"] = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = "roc_auc" if "roc_auc" in df.columns else ("accuracy" if "accuracy" in df.columns else "r2")
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df.index += 1

    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        t = Table(title="Model Comparison")
        for col in df.columns:
            t.add_column(col, style="bold cyan" if col == "algorithm" else "white")
        for _, row in df.iterrows():
            t.add_row(*[f"{v:.4f}" if isinstance(v, float) else ("—" if v is None else str(v)) for v in row.values])
        console.print(t)
    except ImportError:
        print(df.to_string())
    return df


__all__ = ["Evaluator", "EvaluationReport", "compare"]
