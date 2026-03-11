"""
pycheron.infer — Serve trained models as REST APIs.

Usage:
    import pycheron as pycrn

    model = pycrn.train("data.csv", target="label")
    pycrn.infer.serve(model, port=8000)

    # Then POST to:
    # http://localhost:8000/predict  {"data": [[1.2, 3.4, ...]]}
    # http://localhost:8000/health
    # http://localhost:8000/info
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pycheron.train.trained_model import TrainedModel

from pycheron.utils.logging import get_logger

logger = get_logger(__name__)


def serve(
    model: "TrainedModel",
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
) -> None:
    """
    Serve a trained pycheron model as a REST API using Flask.

    Endpoints
    ---------
    POST /predict      — predict from JSON payload {"data": [[...], ...]}
    POST /predict_proba — class probabilities (classification only)
    GET  /health       — health check → {"status": "ok"}
    GET  /info         — model metadata → algorithm, task, version

    Parameters
    ----------
    model   : TrainedModel to serve
    host    : bind address (default '0.0.0.0')
    port    : port number (default 8000)
    workers : number of worker processes
    reload  : auto-reload on code changes (dev only)

    Examples
    --------
    >>> pycrn.infer.serve(model, port=8000)

    Then call:
    >>> import requests
    >>> r = requests.post("http://localhost:8000/predict",
    ...                   json={"data": [[1.2, 3.4, 5.6]]})
    >>> print(r.json())
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise ImportError(
            "Flask is required for model serving: pip install flask"
        )

    import numpy as np
    import pandas as pd

    app = Flask("pycheron-serve")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": model.algorithm})

    @app.route("/info", methods=["GET"])
    def info():
        return jsonify({
            "algorithm": model.algorithm,
            "task": model.task.value,
            "target": model.target,
            "n_features": len(model.feature_names),
            "feature_names": model.feature_names,
            "pycheron_version": _get_version(),
        })

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            payload = request.get_json(force=True)
            X = _parse_payload(payload, model.feature_names)
            preds = model.estimator.predict(X)
            preds = model.preprocessor.decode_target(preds)
            return jsonify({
                "predictions": preds.tolist(),
                "n_samples": len(preds),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/predict_proba", methods=["POST"])
    def predict_proba():
        if not model.task.is_classification:
            return jsonify({"error": "predict_proba only for classification"}), 400
        if not hasattr(model.estimator, "predict_proba"):
            return jsonify({"error": f"{model.algorithm} does not support predict_proba"}), 400
        try:
            payload = request.get_json(force=True)
            X = _parse_payload(payload, model.feature_names)
            proba = model.estimator.predict_proba(X)
            return jsonify({
                "probabilities": proba.tolist(),
                "n_samples": len(proba),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    logger.info(f"Starting pycheron model server on http://{host}:{port}")
    logger.info(f"  Model: {model.algorithm} | Task: {model.task.value}")
    logger.info(f"  POST http://{host}:{port}/predict")
    logger.info(f"  GET  http://{host}:{port}/health")
    logger.info(f"  GET  http://{host}:{port}/info")

    app.run(host=host, port=port, debug=reload)


def _parse_payload(payload: dict, feature_names: list):
    """Parse JSON payload into numpy array."""
    import numpy as np
    import pandas as pd

    if "data" not in payload:
        raise ValueError("Payload must have a 'data' key: {'data': [[...], ...]}")

    data = payload["data"]
    if isinstance(data, list) and len(data) > 0:
        if not isinstance(data[0], list):
            data = [data]  # single sample

    if feature_names:
        df = pd.DataFrame(data, columns=feature_names[:len(data[0])])
        return df.values
    return np.array(data)


def _get_version() -> str:
    try:
        from pycheron._version import __version__
        return __version__
    except Exception:
        return "unknown"


def predict_from_file(
    model: "TrainedModel",
    path: str,
    output_path: Optional[str] = None,
) -> "pd.DataFrame":
    """
    Batch predict from a file and optionally save results.

    Parameters
    ----------
    model       : trained pycheron model
    path        : input CSV/Parquet file
    output_path : save predictions here (CSV) if given

    Returns
    -------
    DataFrame with original data + 'prediction' column

    Examples
    --------
    >>> results = pycrn.infer.predict_from_file(model, "new_data.csv")
    """
    import pandas as pd
    from pycheron.data.loader import DataLoader

    df = DataLoader().load(path)
    X = model.preprocessor.transform(df)
    preds = model.estimator.predict(X)
    preds = model.preprocessor.decode_target(preds)

    df = df.copy()
    df["prediction"] = preds

    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    return df
