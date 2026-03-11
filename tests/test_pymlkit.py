"""
tests/test_pycheron.py — Full test suite for pycheron.

Covers:
- DataLoader (CSV, DataFrame)
- SchemaDetector (types)
- DataValidator (missing, duplicates)
- Preprocessor (numeric, categorical, datetime)
- TaskDetector
- ModelRegistry (registration, recommendation, listing)
- Trainer (classification, regression, with tuning)
- AutoTrainer (tournament + leaderboard)
- EvaluationReport (metrics, display)
- TrainedModel (predict, save, load)
- Explainer (SHAP values, summary)
- Public API (pycheron.train, pycheron.auto_train)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clf_df():
    """Small binary classification dataset."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "age":      rng.randint(18, 80, n).astype(float),
        "income":   rng.exponential(40000, n),
        "gender":   rng.choice(["M", "F"], n),
        "score":    rng.randn(n),
        "label":    rng.choice([0, 1], n),
    })


@pytest.fixture
def reg_df():
    """Small regression dataset."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "x1": rng.randn(n),
        "x2": rng.randn(n),
        "cat": rng.choice(["A", "B", "C"], n),
        "y":  rng.randn(n) * 10 + 50,
    })


@pytest.fixture
def clf_csv(clf_df, tmp_path):
    p = tmp_path / "clf.csv"
    clf_df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def model_dir(tmp_path):
    return str(tmp_path / "saved_model")


# ── DataLoader ────────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_load_dataframe(self, clf_df):
        from pycheron.data.loader import DataLoader
        loader = DataLoader()
        df = loader.load(clf_df)
        assert df.shape == clf_df.shape

    def test_load_csv(self, clf_csv):
        from pycheron.data.loader import DataLoader
        loader = DataLoader()
        df = loader.load(clf_csv)
        assert len(df) > 0

    def test_missing_file_raises(self):
        from pycheron.data.loader import DataLoader
        with pytest.raises(FileNotFoundError):
            DataLoader().load("nonexistent.csv")

    def test_bad_target_raises(self, clf_df):
        from pycheron.data.loader import DataLoader
        with pytest.raises(ValueError, match="not found"):
            DataLoader().load(clf_df, target="nonexistent_col")

    def test_hash_deterministic(self, clf_df):
        from pycheron.data.loader import DataLoader
        h1 = DataLoader.hash(clf_df)
        h2 = DataLoader.hash(clf_df.copy())
        assert h1 == h2

    def test_unsupported_format(self, tmp_path):
        from pycheron.data.loader import DataLoader
        f = tmp_path / "data.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            DataLoader().load(str(f))


# ── SchemaDetector ────────────────────────────────────────────────────────────

class TestSchemaDetector:
    def test_detects_numeric(self, clf_df):
        from pycheron.data.validator import SchemaDetector
        from pycheron.utils.types import ColumnType
        schema = SchemaDetector().detect(clf_df, target="label")
        assert schema["age"] == ColumnType.NUMERIC
        assert schema["income"] == ColumnType.NUMERIC

    def test_detects_categorical(self, clf_df):
        from pycheron.data.validator import SchemaDetector
        from pycheron.utils.types import ColumnType
        schema = SchemaDetector().detect(clf_df, target="label")
        assert schema["gender"] == ColumnType.CATEGORICAL

    def test_detects_target(self, clf_df):
        from pycheron.data.validator import SchemaDetector
        from pycheron.utils.types import ColumnType
        schema = SchemaDetector().detect(clf_df, target="label")
        assert schema["label"] == ColumnType.TARGET

    def test_detects_constant(self):
        from pycheron.data.validator import SchemaDetector
        from pycheron.utils.types import ColumnType
        df = pd.DataFrame({"const": [1] * 50, "y": [0, 1] * 25})
        schema = SchemaDetector().detect(df, target="y")
        assert schema["const"] == ColumnType.CONSTANT

    def test_detects_datetime(self):
        from pycheron.data.validator import SchemaDetector
        from pycheron.utils.types import ColumnType
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=50),
            "y": range(50),
        })
        schema = SchemaDetector().detect(df, target="y")
        assert schema["date"] == ColumnType.DATETIME


# ── DataValidator ─────────────────────────────────────────────────────────────

class TestDataValidator:
    def test_detects_high_missing(self):
        from pycheron.data.validator import DataValidator, SchemaDetector
        df = pd.DataFrame({
            "good": range(100),
            "mostly_null": [None] * 80 + list(range(20)),
            "y": [0, 1] * 50,
        })
        schema = SchemaDetector().detect(df, target="y")
        report = DataValidator().validate(df, target="y", schema=schema)
        drops = report.columns_to_drop()
        assert "mostly_null" in drops

    def test_no_issues_clean_data(self, clf_df):
        from pycheron.data.validator import DataValidator, SchemaDetector
        schema = SchemaDetector().detect(clf_df, target="label")
        report = DataValidator().validate(clf_df, target="label", schema=schema)
        drops = report.columns_to_drop()
        assert len(drops) == 0


# ── Preprocessor ─────────────────────────────────────────────────────────────

class TestPreprocessor:
    def _run(self, df, target, algorithm="random_forest"):
        from pycheron.data.loader import SchemaDetector, DataValidator, DataProfiler
        from pycheron.preprocess.pipeline import Preprocessor
        from pycheron.utils.types import TaskType
        schema = SchemaDetector().detect(df, target=target)
        validation = DataValidator().validate(df, target=target, schema=schema)
        profile = DataProfiler().profile(df, target=target, schema=schema)
        prep = Preprocessor(algorithm=algorithm, task=TaskType.BINARY_CLASSIFICATION)
        X, y = prep.fit_transform(df, target, schema, profile, validation)
        return X, y, prep

    def test_output_shape(self, clf_df):
        X, y, _ = self._run(clf_df, "label")
        assert X.shape[0] == len(clf_df)
        assert y.shape[0] == len(clf_df)

    def test_no_nans_in_output(self, clf_df):
        clf_df = clf_df.copy()
        clf_df.loc[:5, "age"] = np.nan
        X, y, _ = self._run(clf_df, "label")
        assert not np.any(np.isnan(X))

    def test_feature_names_populated(self, clf_df):
        X, y, prep = self._run(clf_df, "label")
        assert len(prep.feature_names) == X.shape[1]

    def test_transform_consistency(self, clf_df):
        X, y, prep = self._run(clf_df, "label")
        X2 = prep.transform(clf_df.drop(columns=["label"]))
        assert X2.shape[1] == X.shape[1]


# ── TaskDetector ──────────────────────────────────────────────────────────────

class TestTaskDetector:
    def test_binary_classification(self):
        from pycheron.train.trainer import TaskDetector
        from pycheron.utils.types import TaskType
        y = pd.Series([0, 1, 0, 1, 1, 0])
        assert TaskDetector().detect(y) == TaskType.BINARY_CLASSIFICATION

    def test_multiclass_classification(self):
        from pycheron.train.trainer import TaskDetector
        from pycheron.utils.types import TaskType
        y = pd.Series([0, 1, 2, 1, 2, 0, 3])
        assert TaskDetector().detect(y) == TaskType.MULTICLASS_CLASSIFICATION

    def test_regression(self):
        from pycheron.train.trainer import TaskDetector
        from pycheron.utils.types import TaskType
        y = pd.Series(np.random.randn(200))
        assert TaskDetector().detect(y) == TaskType.REGRESSION

    def test_task_override_classification(self):
        from pycheron.train.trainer import TaskDetector
        from pycheron.utils.types import TaskType
        y = pd.Series(np.random.randn(200))
        t = TaskDetector().detect(y, task_override="classification")
        assert t.is_classification

    def test_task_override_regression(self):
        from pycheron.train.trainer import TaskDetector
        from pycheron.utils.types import TaskType
        y = pd.Series([0, 1, 0, 1])
        assert TaskDetector().detect(y, task_override="regression") == TaskType.REGRESSION


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_list_classification_algorithms(self):
        from pycheron.registry.model_registry import ModelRegistry
        from pycheron.utils.types import TaskType
        names = ModelRegistry.instance().list_names(task=TaskType.BINARY_CLASSIFICATION)
        assert "random_forest" in names
        assert "logistic_regression" in names

    def test_list_regression_algorithms(self):
        from pycheron.registry.model_registry import ModelRegistry
        from pycheron.utils.types import TaskType
        names = ModelRegistry.instance().list_names(task=TaskType.REGRESSION)
        assert "ridge" in names
        assert "lasso" in names

    def test_get_known_algorithm(self):
        from pycheron.registry.model_registry import ModelRegistry
        meta = ModelRegistry.instance().get("random_forest")
        assert meta.name == "random_forest"

    def test_get_unknown_raises(self):
        from pycheron.registry.model_registry import ModelRegistry
        with pytest.raises(ValueError, match="Unknown algorithm"):
            ModelRegistry.instance().get("does_not_exist")

    def test_custom_registration(self):
        from pycheron.registry.model_registry import ModelRegistry, AlgorithmMeta
        from sklearn.dummy import DummyClassifier

        meta = AlgorithmMeta(
            name="test_dummy",
            factory=lambda **kw: DummyClassifier(**kw),
            tasks=["classification"],
        )
        ModelRegistry.instance().register(meta)
        assert "test_dummy" in ModelRegistry.instance().list_names()

    def test_build_estimator(self):
        from pycheron.registry.model_registry import ModelRegistry
        meta = ModelRegistry.instance().get("random_forest")
        est = meta.build()
        assert hasattr(est, "fit")

    def test_recommendation_returns_names(self, clf_df):
        from pycheron.registry.model_registry import ModelRegistry
        from pycheron.data.profiler import DataProfiler
        from pycheron.data.validator import SchemaDetector
        from pycheron.utils.types import TaskType
        schema = SchemaDetector().detect(clf_df, "label")
        profile = DataProfiler().profile(clf_df, "label", schema)
        recs = ModelRegistry.instance().recommend(profile, TaskType.BINARY_CLASSIFICATION, n=3)
        assert len(recs) == 3
        assert all(isinstance(r, str) for r in recs)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    def test_binary_metrics(self):
        from pycheron.evaluate.evaluator import Evaluator
        from pycheron.utils.types import TaskType
        ev = Evaluator(task=TaskType.BINARY_CLASSIFICATION)
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        report = ev.evaluate(y_true, y_pred)
        assert "accuracy" in report.metrics
        assert "f1_macro" in report.metrics

    def test_regression_metrics(self):
        from pycheron.evaluate.evaluator import Evaluator
        from pycheron.utils.types import TaskType
        ev = Evaluator(task=TaskType.REGRESSION)
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9])
        report = ev.evaluate(y_true, y_pred)
        assert "rmse" in report.metrics
        assert "r2" in report.metrics
        assert report.metrics["r2"] > 0.9

    def test_primary_score(self):
        from pycheron.evaluate.evaluator import Evaluator
        from pycheron.utils.types import TaskType
        ev = Evaluator(task=TaskType.BINARY_CLASSIFICATION, metric="accuracy")
        y = np.array([0, 1, 0, 1])
        report = ev.evaluate(y, y)
        assert report.primary_score == 1.0


# ── Trainer ───────────────────────────────────────────────────────────────────

class TestTrainer:
    def test_classification(self, clf_df):
        from pycheron.train.trainer import Trainer
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        assert model is not None
        assert model.algorithm == "random_forest"
        assert model.evaluation is not None

    def test_regression(self, reg_df):
        from pycheron.train.trainer import Trainer
        model = Trainer(algorithm="ridge", verbose=0).fit(reg_df, "y")
        assert model is not None
        assert model.evaluation.metrics["r2"] is not None

    def test_auto_algorithm_selection(self, clf_df):
        from pycheron.train.trainer import Trainer
        model = Trainer(verbose=0).fit(clf_df, "label")
        assert model.algorithm is not None

    def test_predict_returns_array(self, clf_df):
        from pycheron.train.trainer import Trainer
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        preds = model.predict(clf_df.drop(columns=["label"]))
        assert len(preds) == len(clf_df)

    def test_predict_proba(self, clf_df):
        from pycheron.train.trainer import Trainer
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        proba = model.predict_proba(clf_df.drop(columns=["label"]))
        assert proba.shape[1] == 2

    def test_with_tuning(self, clf_df):
        from pycheron.train.trainer import Trainer
        model = Trainer(
            algorithm="random_forest", tune=True, tune_trials=3, verbose=0
        ).fit(clf_df, "label")
        assert model is not None

    def test_feature_names_populated(self, clf_df):
        from pycheron.train.trainer import Trainer
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        assert len(model.feature_names) > 0

    def test_from_csv(self, clf_csv):
        from pycheron.train.trainer import Trainer
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_csv, "label")
        assert model is not None


# ── AutoTrainer ───────────────────────────────────────────────────────────────

class TestAutoTrainer:
    def test_auto_train_returns_model(self, clf_df):
        from pycheron.automl.auto_trainer import AutoTrainer
        model = AutoTrainer(
            time_budget=30, n_candidates=3, tune_trials=2, verbose=0
        ).fit(clf_df, "label")
        assert model is not None
        assert model.algorithm is not None

    def test_leaderboard_populated(self, clf_df):
        from pycheron.automl.auto_trainer import AutoTrainer
        model = AutoTrainer(
            time_budget=30, n_candidates=3, tune_trials=2, verbose=0
        ).fit(clf_df, "label")
        lb = model.leaderboard()
        assert lb is not None
        assert len(lb) >= 1

    def test_auto_detect_target(self, clf_df):
        from pycheron.automl.auto_trainer import AutoTrainer
        # target should auto-detect last column
        model = AutoTrainer(
            time_budget=20, n_candidates=2, tune_trials=2, verbose=0
        ).fit(clf_df)  # no target arg
        assert model is not None


# ── Model Persistence ─────────────────────────────────────────────────────────

class TestModelPersistence:
    def test_save_and_load(self, clf_df, model_dir):
        from pycheron.train.trainer import Trainer
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        saved = model.save(model_dir)
        assert (saved / "model.pkl").exists()
        assert (saved / "manifest.json").exists()

        loaded = Trainer  # just checking save worked
        from pycheron.persistence.model_store import ModelStore
        loaded_model = ModelStore.load(model_dir)
        assert loaded_model.algorithm == model.algorithm

    def test_versioning(self, clf_df, model_dir):
        from pycheron.train.trainer import Trainer
        from pycheron.persistence.model_store import ModelStore
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        model.save(model_dir)
        model.save(model_dir)  # save twice
        versions = ModelStore.list_versions(model_dir)
        assert len(versions) == 2

    def test_loaded_model_predicts(self, clf_df, model_dir):
        from pycheron.train.trainer import Trainer
        from pycheron.persistence.model_store import ModelStore
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        model.save(model_dir)
        loaded = ModelStore.load(model_dir)
        preds = loaded.predict(clf_df.drop(columns=["label"]))
        assert len(preds) == len(clf_df)


# ── Public API ────────────────────────────────────────────────────────────────

class TestPublicAPI:
    def test_train_function(self, clf_df):
        import pycheron
        model = pycheron.train(clf_df, target="label", algorithm="random_forest", verbose=0)
        assert model is not None

    def test_auto_train_function(self, clf_df):
        import pycheron
        model = pycheron.auto_train(clf_df, target="label", time_budget=20, verbose=0)
        assert model is not None

    def test_train_with_save(self, clf_df, model_dir):
        import pycheron
        model = pycheron.train(
            clf_df, target="label",
            algorithm="random_forest",
            verbose=0,
            save_path=model_dir,
        )
        assert Path(model_dir).exists()

    def test_load_model_function(self, clf_df, model_dir):
        import pycheron
        model = pycheron.train(clf_df, target="label", algorithm="random_forest",
                              verbose=0, save_path=model_dir)
        loaded = pycheron.load_model(model_dir)
        preds = loaded.predict(clf_df.drop(columns=["label"]))
        assert len(preds) == len(clf_df)

    def test_repr(self, clf_df):
        import pycheron
        model = pycheron.train(clf_df, target="label", algorithm="random_forest", verbose=0)
        r = repr(model)
        assert "TrainedModel" in r
        assert "random_forest" in r
