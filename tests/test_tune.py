"""Tests for src/tune.py helpers and for the saved final model (models/ is created by python src/tune.py)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from load_data import FEATURES, get_splits
from tune import META_PATH, MODEL_PATH, build_xgb_pipeline, expected_calibration_error, full_metrics, suggest_params

saved_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="run python src/tune.py first")


def test_search_space_stays_in_range():
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    for _ in range(20):
        p = suggest_params(study.ask())
        assert 0.01 <= p["learning_rate"] <= 0.3
        assert 3 <= p["max_depth"] <= 10
        assert 1 <= p["min_child_weight"] <= 20
        assert 0.6 <= p["subsample"] <= 1.0 and 0.5 <= p["colsample_bytree"] <= 1.0
        assert 100 <= p["n_estimators"] <= 1000 and p["n_estimators"] % 50 == 0
        assert 1e-3 <= p["reg_lambda"] <= 10 and 0 <= p["gamma"] <= 5
    assert len(p) == 8


def test_pipeline_uses_scale_pos_weight_and_fixed_settings():
    import pandas as pd
    y = pd.Series([0, 0, 0, 1])
    model = build_xgb_pipeline({"max_depth": 3}, y).named_steps["model"]
    assert model.scale_pos_weight == pytest.approx(3.0)
    assert model.n_jobs == 2 and model.random_state == 42 and model.max_depth == 3


def test_expected_calibration_error():
    y = np.array([0] * 7 + [1] * 3)
    assert expected_calibration_error(y, np.full(10, 0.3)) == pytest.approx(0.0)   # 30% predicted, 30% default
    assert expected_calibration_error(y, np.full(10, 0.8)) == pytest.approx(0.5)   # over-predicts by 50 points


def test_full_metrics_has_brier_and_ece():
    m = full_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.7]))
    assert m["brier"] == pytest.approx(np.mean(np.square([0.1, 0.1, 0.2, 0.3])))
    assert {"roc_auc", "pr_auc", "ks", "precision", "recall", "f1", "brier", "ece"} == set(m)


@saved_model
def test_meta_file_is_complete():
    meta = json.loads(META_PATH.read_text())
    for key in ["model_type", "version", "training_date", "features", "numeric_features", "categorical_features",
                "best_params", "test", "versions", "calibrated"]:
        assert key in meta
    assert meta["features"] == FEATURES
    assert {"scikit-learn", "xgboost"} <= set(meta["versions"])
    assert 0.85 <= meta["test"]["roc_auc"] <= 0.97          # target met, below the leakage alarm
    assert np.array(meta["test"]["confusion_matrix"]).sum() == meta["n_test"]


@saved_model
def test_saved_model_scores_raw_rows_with_missing_and_unseen_values():
    model = joblib.load(MODEL_PATH)
    X = get_splits()["X_val"].head(5).copy()
    X.loc[X.index[0], ["loan_int_rate", "person_emp_length"]] = np.nan
    X.loc[X.index[1], "loan_intent"] = "SPACE_TRAVEL"
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()
