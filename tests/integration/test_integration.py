"""
tests/integration/test_integration.py — End-to-end integration tests.

These tests exercise the full pipeline from raw data to trained model,
covering train(), auto_train(), save/load, and the public API.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pymlkit
from pymlkit.automl.auto_trainer import AutoTrainer
from pymlkit.persistence.model_store import ModelStore
from pymlkit.train.trainer import TaskDetector, Trainer
from pymlkit.utils.types import TaskType


# ── TaskDetector ──────────────────────────────────────────────────────────────

class TestTaskDetector:
    def test_binary(self):
        assert TaskDetector().detect(pd.Series([0, 1, 0, 1])) == TaskType.BINARY_CLASSIFICATION

    def test_multiclass(self):
        assert TaskDetector().detect(pd.Series([0, 1, 2, 3])) == TaskType.MULTICLASS_CLASSIFICATION

    def test_regression(self):
        y = pd.Series(np.random.randn(200))
        assert TaskDetector().detect(y) == TaskType.REGRESSION

    def test_override_classification(self):
        y = pd.Series(np.random.randn(200))
        assert TaskDetector().detect(y, task_override="classification").is_classification

    def test_override_regression(self):
        assert TaskDetector().detect(pd.Series([0, 1]), task_override="regression") == TaskType.REGRESSION


# ── Trainer ───────────────────────────────────────────────────────────────────

class TestTrainer:
    def test_classification(self, clf_df):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        assert model.algorithm == "random_forest"
        assert model.evaluation is not None

    def test_regression(self, reg_df):
        model = Trainer(algorithm="ridge", verbose=0).fit(reg_df, "y")
        assert model.evaluation.metrics["r2"] is not None

    def test_auto_algorithm_selection(self, clf_df):
        model = Trainer(verbose=0).fit(clf_df, "label")
        assert model.algorithm is not None

    def test_predict_shape(self, clf_df):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        preds = model.predict(clf_df.drop(columns=["label"]))
        assert len(preds) == len(clf_df)

    def test_predict_proba_shape(self, clf_df):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        proba = model.predict_proba(clf_df.drop(columns=["label"]))
        assert proba.shape[1] == 2

    def test_tuning(self, clf_df):
        model = Trainer(algorithm="random_forest", tune=True, tune_trials=3, verbose=0).fit(clf_df, "label")
        assert model is not None

    def test_feature_names_populated(self, clf_df):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        assert len(model.feature_names) > 0

    def test_from_csv(self, clf_csv):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_csv, "label")
        assert model is not None


# ── AutoTrainer ───────────────────────────────────────────────────────────────

class TestAutoTrainer:
    def test_returns_model(self, clf_df):
        model = AutoTrainer(time_budget=30, n_candidates=3, tune_trials=2, verbose=0).fit(clf_df, "label")
        assert model.algorithm is not None

    def test_leaderboard_populated(self, clf_df):
        model = AutoTrainer(time_budget=30, n_candidates=3, tune_trials=2, verbose=0).fit(clf_df, "label")
        lb = model.leaderboard()
        assert lb is not None
        assert len(lb) >= 1

    def test_auto_detect_target(self, clf_df):
        model = AutoTrainer(time_budget=20, n_candidates=2, tune_trials=2, verbose=0).fit(clf_df)
        assert model is not None


# ── Model Persistence ─────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_creates_files(self, clf_df, model_dir):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        saved = model.save(model_dir)
        assert (saved / "model.pkl").exists()
        assert (saved / "manifest.json").exists()

    def test_load_returns_same_algorithm(self, clf_df, model_dir):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        model.save(model_dir)
        loaded = ModelStore.load(model_dir)
        assert loaded.algorithm == model.algorithm

    def test_loaded_model_predicts(self, clf_df, model_dir):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        model.save(model_dir)
        loaded = ModelStore.load(model_dir)
        preds = loaded.predict(clf_df.drop(columns=["label"]))
        assert len(preds) == len(clf_df)

    def test_versioning(self, clf_df, model_dir):
        model = Trainer(algorithm="random_forest", verbose=0).fit(clf_df, "label")
        model.save(model_dir)
        model.save(model_dir)
        versions = ModelStore.list_versions(model_dir)
        assert len(versions) == 2


# ── Public API ────────────────────────────────────────────────────────────────

class TestPublicAPI:
    def test_train(self, clf_df):
        model = pymlkit.train(clf_df, target="label", algorithm="random_forest", verbose=0)
        assert model is not None

    def test_auto_train(self, clf_df):
        model = pymlkit.auto_train(clf_df, target="label", time_budget=20, n_candidates=2, verbose=0)
        assert model is not None

    def test_train_with_save(self, clf_df, model_dir):
        pymlkit.train(clf_df, target="label", algorithm="random_forest", verbose=0, save_path=model_dir)
        assert Path(model_dir).exists()

    def test_load_model(self, clf_df, model_dir):
        pymlkit.train(clf_df, target="label", algorithm="random_forest", verbose=0, save_path=model_dir)
        loaded = pymlkit.load_model(model_dir)
        preds = loaded.predict(clf_df.drop(columns=["label"]))
        assert len(preds) == len(clf_df)

    def test_repr(self, clf_df):
        model = pymlkit.train(clf_df, target="label", algorithm="random_forest", verbose=0)
        r = repr(model)
        assert "TrainedModel" in r
        assert "random_forest" in r
