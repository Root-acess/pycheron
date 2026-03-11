"""
pycheron.tracking — Experiment tracking and run leaderboard.

Usage:
    import pycheron as pycrn

    pycrn.tracking.log("my_experiment", model)
    pycrn.tracking.leaderboard()
    pycrn.tracking.clear()
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

import pandas as pd

from pycheron.utils.logging import get_logger

if TYPE_CHECKING:
    from pycheron.train.trained_model import TrainedModel

logger = get_logger(__name__)

_DEFAULT_LOG_FILE = Path.home() / ".pycheron" / "runs.json"


def _ensure_log_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]")


def log(
    name: str,
    model: "TrainedModel",
    tags: Optional[Dict[str, str]] = None,
    notes: str = "",
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Log a trained model run to the experiment tracker.

    Parameters
    ----------
    name     : experiment name (e.g. 'titanic_v1')
    model    : TrainedModel from pycrn.train() or pycrn.auto_train()
    tags     : optional dict of metadata tags
    notes    : free-text notes
    log_file : custom path for the runs log file

    Returns
    -------
    dict of the logged run record

    Examples
    --------
    >>> pycrn.tracking.log("experiment_1", model, tags={"dataset": "titanic"})
    """
    log_path = Path(log_file) if log_file else _DEFAULT_LOG_FILE
    _ensure_log_file(log_path)

    metrics = {}
    primary_metric = ""
    primary_score = 0.0
    if model.evaluation:
        metrics = model.evaluation.metrics
        primary_metric = model.evaluation.primary_metric
        primary_score = model.evaluation.primary_score

    run = {
        "id": f"run_{int(time.time() * 1000)}",
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "algorithm": model.algorithm,
        "task": model.task.value,
        "target": model.target,
        "n_features": len(model.feature_names),
        "primary_metric": primary_metric,
        "primary_score": round(primary_score, 6),
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "params": model.meta.get("best_params", {}),
        "training_time": round(model.meta.get("training_time", 0), 2),
        "n_rows": model.meta.get("n_rows", 0),
        "tags": tags or {},
        "notes": notes,
    }

    # Load existing runs, append, save
    existing = json.loads(log_path.read_text())
    existing.append(run)
    log_path.write_text(json.dumps(existing, indent=2))

    logger.info(
        f"Tracked run '{name}' — {primary_metric}={primary_score:.4f} "
        f"[{model.algorithm}] → {log_path}"
    )
    return run


def leaderboard(
    log_file: Optional[Path] = None,
    sort_by: Optional[str] = None,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Display all tracked runs as a sorted leaderboard.

    Parameters
    ----------
    log_file : custom log file path
    sort_by  : metric to sort by (uses primary_score if None)
    top_n    : show only top N runs

    Returns
    -------
    pd.DataFrame of all runs

    Examples
    --------
    >>> pycrn.tracking.leaderboard()
    >>> pycrn.tracking.leaderboard(top_n=5)
    """
    log_path = Path(log_file) if log_file else _DEFAULT_LOG_FILE

    if not log_path.exists():
        logger.info("No experiment runs logged yet. Use pycrn.tracking.log() first.")
        return pd.DataFrame()

    runs = json.loads(log_path.read_text())
    if not runs:
        logger.info("No runs found in the log file.")
        return pd.DataFrame()

    rows = []
    for r in runs:
        row = {
            "name": r["name"],
            "algorithm": r["algorithm"],
            "task": r["task"],
            "primary_metric": r["primary_metric"],
            "score": r["primary_score"],
            "training_time(s)": r["training_time"],
            "n_rows": r["n_rows"],
            "timestamp": r["timestamp"][:19],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_col = sort_by if sort_by and sort_by in df.columns else "score"
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    df.index += 1

    if top_n:
        df = df.head(top_n)

    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        t = Table(title=f"Experiment Leaderboard ({len(df)} runs)")
        for col in df.columns:
            style = "bold cyan" if col == "name" else ("bold green" if col == "score" else "white")
            t.add_column(col, style=style)
        for _, row in df.iterrows():
            t.add_row(*[
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in row.values
            ])
        console.print(t)
    except ImportError:
        print(df.to_string())

    return df


def clear(log_file: Optional[Path] = None) -> None:
    """
    Clear all tracked experiment runs.

    Examples
    --------
    >>> pycrn.tracking.clear()
    """
    log_path = Path(log_file) if log_file else _DEFAULT_LOG_FILE
    if log_path.exists():
        log_path.write_text("[]")
        logger.info(f"Cleared all runs from {log_path}")
    else:
        logger.info("No log file found to clear.")


def get_run(name: str, log_file: Optional[Path] = None) -> Optional[Dict]:
    """Return the most recent run with the given name."""
    log_path = Path(log_file) if log_file else _DEFAULT_LOG_FILE
    if not log_path.exists():
        return None
    runs = json.loads(log_path.read_text())
    matches = [r for r in runs if r["name"] == name]
    return matches[-1] if matches else None
