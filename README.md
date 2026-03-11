# pycheron

> **The all-in-one Python ML framework.**  
> Train, clean, visualize, track, and serve — from a single import.

```bash
pip install pycheron
```

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.3-blue?style=flat-square&logo=python)](https://pypi.org/project/pycheron)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

---

## Why pycheron?

pycheron wraps pandas, numpy, and sklearn — adding ML-specific helpers — so you never need to juggle multiple libraries. One import gives you the full stack.

```python
import pycheron as pycrn   # just like: import pandas as pd
```

---

## Quick Start

### Load & Clean

```python
import pycheron as pycrn

# Auto-detect CSV, Parquet, JSON, or Excel
df = pycrn.pd.read_smart("data.csv")

# Extended describe() — includes missing%, skew, cardinality
pycrn.pd.summary(df)

# Auto-clean: fills nulls, removes dupes, warns on class imbalance
df_clean = pycrn.pd.clean(df)

# Profile the dataset
pycrn.data.profile(df_clean, target="label")

# Stratified train/test split
train_df, test_df = pycrn.pd.split(df_clean, target="label")
```

### Train

```python
# One-line training — auto-selects best algorithm
model = pycrn.train("data.csv", target="label")

# With options
model = pycrn.train(
    df_clean, target="label",
    algorithm="random_forest",
    tune=True, tune_trials=50,
    cv=5, test_size=0.2
)

# Full AutoML tournament (tries 6 algorithms, tunes the winner)
model = pycrn.auto_train("data.csv", time_budget=120)
model.leaderboard()
```

### Predict & Evaluate

```python
predictions = model.predict("test.csv")
proba       = model.predict_proba("test.csv")   # classification only

model.evaluate()   # prints full metrics table

# Compare multiple models
m1 = pycrn.train(df, target="label", algorithm="random_forest")
m2 = pycrn.train(df, target="label", algorithm="gradient_boosting")
pycrn.evaluate.compare([m1, m2], X_test, y_test)
```

### Visualize

```python
pycrn.plot.feature_importance(model, top_n=15)
pycrn.plot.confusion_matrix(model, X_test, y_test)
pycrn.plot.learning_curve(model, X_train, y_train)
pycrn.plot.distribution(df, column="age", hue="label")
pycrn.plot.correlation(df)
pycrn.plot.class_balance(df, target="label")
pycrn.plot.shap_summary(model, X_test)
```

### Track Experiments

```python
pycrn.tracking.log("titanic_v1", model, tags={"dataset": "titanic"})
pycrn.tracking.leaderboard()   # shows all logged runs sorted by score
pycrn.tracking.clear()         # reset
```

### Serve as REST API

```python
# Requires: pip install "pycheron[serve]"
pycrn.infer.serve(model, port=8000)

# Endpoints:
# POST http://localhost:8000/predict       {"data": [[1.2, 3.4, ...]]}
# POST http://localhost:8000/predict_proba {"data": [...]}
# GET  http://localhost:8000/health
# GET  http://localhost:8000/info
```

### Save & Load

```python
model.save("./my_model")
model = pycrn.load_model("./my_model")
```

---

## Installation

```bash
# Core
pip install pycheron

# With all extras (XGBoost, LightGBM, PyTorch, Flask)
pip install "pycheron[all]"

# Individual extras
pip install "pycheron[xgboost]"
pip install "pycheron[lightgbm]"
pip install "pycheron[serve]"    # for pycrn.infer.serve()
```

### Requirements
- Python 3.9+
- numpy, pandas, scikit-learn, shap, optuna, rich, matplotlib (all auto-installed)

---

## Wrapped Libraries

pycheron exposes pandas and numpy through its own namespace with added ML helpers:

### pycrn.pd (pandas + helpers)

| Function | Description |
|---|---|
| `pycrn.pd.read_smart(path)` | Auto-detect and load CSV, Parquet, JSON, Excel |
| `pycrn.pd.clean(df)` | Auto-fill nulls, remove dupes, drop high-missing columns |
| `pycrn.pd.split(df, target)` | Stratified train/val/test split |
| `pycrn.pd.summary(df)` | Extended describe() with missing%, skew, cardinality |
| `pycrn.pd.profile(df, target)` | Rich statistical profile table |
| All pandas functions | `pycrn.pd.read_csv()`, `.merge()`, `.concat()`, etc. |

### pycrn.np (numpy + helpers)

| Function | Description |
|---|---|
| `pycrn.np.normalize(arr, method)` | 'minmax', 'zscore', or 'l2' normalization |
| `pycrn.np.to_frame(arr, columns)` | Convert numpy array to pandas DataFrame |
| `pycrn.np.clip_outliers(arr)` | IQR or z-score outlier clipping |
| `pycrn.np.describe(arr)` | Summary statistics dict for an array |
| All numpy functions | `pycrn.np.array()`, `.zeros()`, `.linalg`, etc. |

---

## Supported Algorithms

| Algorithm | Classification | Regression |
|---|:---:|:---:|
| `random_forest` | ✅ | ✅ |
| `gradient_boosting` | ✅ | ✅ |
| `logistic_regression` | ✅ | — |
| `ridge` | — | ✅ |
| `lasso` | — | ✅ |
| `svm` | ✅ | ✅ |
| `knn` | ✅ | ✅ |
| `decision_tree` | ✅ | ✅ |
| `xgboost` *(optional)* | ✅ | ✅ |
| `lightgbm` *(optional)* | ✅ | ✅ |

Leave `algorithm=None` (default) to let pycheron auto-select based on your data profile.

---

## Full API Reference

### Training

```python
pycrn.train(data, target, *, algorithm=None, task=None, test_size=0.2,
            cv=5, tune=False, tune_trials=30, time_budget=None,
            metric=None, random_state=42, n_jobs=-1, verbose=1, save_path=None)

pycrn.auto_train(data, target=None, *, time_budget=120, n_candidates=6,
                 tune_trials=30, metric=None, random_state=42, n_jobs=-1,
                 verbose=1, save_path=None)
```

### TrainedModel methods

```python
model.predict(data)              # → np.ndarray of labels
model.predict_proba(data)        # → np.ndarray of probabilities
model.evaluate()                 # → displays EvaluationReport
model.explain(data, n_samples)   # → SHAP Explanation object
model.save(path)                 # → saves to disk
model.leaderboard()              # → AutoML tournament results (auto_train only)
```

### Data

```python
pycrn.data.load(path, target)
pycrn.data.clean(df, *, drop_duplicates, fill_numeric, fill_categorical,
                 drop_missing_threshold, clip_outliers, verbose)
pycrn.data.profile(df, target)
pycrn.data.validate(df, target)
pycrn.data.split(df, target, *, test_size, val_size, stratify, random_state)
```

### Plot

```python
pycrn.plot.feature_importance(model, top_n, title, figsize, save_path)
pycrn.plot.confusion_matrix(model, X_test, y_test, class_names, normalize)
pycrn.plot.learning_curve(model, X, y, cv, scoring, train_sizes)
pycrn.plot.distribution(df, column, hue, bins)
pycrn.plot.correlation(df, method, annot, mask_upper)
pycrn.plot.class_balance(df, target)
pycrn.plot.shap_summary(model, X, n_samples)
pycrn.plot.shap_waterfall(model, X, sample_idx)
```

### Tracking

```python
pycrn.tracking.log(name, model, tags, notes)
pycrn.tracking.leaderboard(sort_by, top_n)
pycrn.tracking.get_run(name)
pycrn.tracking.clear()
```

### Inference

```python
pycrn.infer.serve(model, host, port, workers, reload)
pycrn.infer.predict_from_file(model, path, output_path)
```

### Evaluate

```python
pycrn.evaluate.compare(models, X_test, y_test)
```

---

## Project Structure

```
pycheron/
├── __init__.py          ← master wiring file
├── _api.py              ← train(), auto_train(), load_model(), explain()
├── _version.py
├── libs/                ← pycrn.pd / pycrn.np / pycrn.sklearn
├── data/                ← load, clean, profile, validate, split
├── preprocess/          ← encoder, scaler, pipeline
├── train/               ← trainer, trained_model
├── automl/              ← auto_trainer (tournament engine)
├── evaluate/            ← evaluator, compare
├── explain/             ← SHAP explainer
├── plot/                ← 8 visualization functions
├── optimize/            ← Optuna tuner
├── registry/            ← algorithm catalogue
├── persistence/         ← model save/load
├── tracking/            ← experiment logger
├── infer/               ← REST API serving
├── config/              ← TOML + env var settings
└── utils/               ← logging, types
```

---

## Changelog

### 0.1.3 — 2026-03-11
- `pycrn.pd` — pandas wrapper with `read_smart()`, `clean()`, `profile()`, `split()`, `summary()`
- `pycrn.np` — numpy wrapper with `normalize()`, `to_frame()`, `clip_outliers()`, `describe()`
- `pycrn.sklearn` — sklearn as a submodule
- `pycrn.plot` — 8 one-line visualization functions
- `pycrn.data.clean()` — AutoCleaner with imbalance detection
- `pycrn.data.split()` — SmartSplitter with val set support
- `pycrn.tracking` — experiment logger + persistent leaderboard
- `pycrn.infer.serve()` — Flask REST API server
- `pycrn.evaluate.compare()` — multi-model comparison table

### 0.1.0 — 2026-03-10
- Initial release with core training, AutoML, SHAP, Optuna tuning

---

## Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install in dev mode: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Submit a pull request

---

## License

MIT © pycheron contributors

---

## Links

- **GitHub**: https://github.com/Root-acess/pycheron
- **PyPI**: https://pypi.org/project/pycheron
- **Issues**: https://github.com/Root-acess/pycheron/issues
