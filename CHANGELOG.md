# Changelog

All notable changes to pycheron will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- `pycheron.tracking` — experiment tracking module
- `pycheron.plugins`  — third-party algorithm plugin system
- `pycheron.infer`    — WSGI-compatible model serving (`serve()`)
- CLI entry point (`pycheron train data.csv --target label`)

---

## [0.1.0] — 2026-03-11

### Added
- `pycheron.train()` — single-call training with auto algorithm selection
- `pycheron.auto_train()` — 3-stage AutoML tournament (sample → CV → tune)
- `pycheron.load_model()` — versioned model persistence via `ModelStore`
- `pycheron.explain()` — SHAP-based feature attribution
- `DataLoader` — CSV / Parquet / JSON / Excel / DataFrame input
- `SchemaDetector` — automatic column type inference (8 types)
- `DataValidator` — data quality checks with `ValidationReport`
- `DataProfiler` — statistical profiling for algorithm recommendation
- `Preprocessor` — intelligent `ColumnTransformer` assembly
  - `encoder.py` — OHE, ordinal, high-cardinality encoding
  - `scaler.py`  — Standard / Robust scaler selection
  - `pipeline.py` — full `ColumnTransformer` orchestration
- `ModelRegistry` — singleton algorithm catalogue with recommendation scoring
- `Evaluator` — automatic metric selection (ROC-AUC, F1, RMSE, R², …)
- `Optimizer` — Optuna Bayesian hyperparameter tuning with graceful fallback
- `Explainer` — SHAP TreeExplainer / LinearExplainer / KernelExplainer wrapper
- `ModelStore` — versioned `model.pkl` + `manifest.json` save/load
- `Config` — TOML + environment variable configuration
- Built-in algorithms: 7 classifiers + 6 regressors (sklearn)
- Optional: XGBoost, LightGBM (auto-registered when installed)
- Restructured codebase into clean, split-module layout
