"""Tests for the model set-up and metrics in src/train.py (small data only - no full training run)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leakage_check import perfect_rule_check
from load_data import get_splits
from train import N_JOBS, build_models, evaluate, ks_statistic


@pytest.fixture(scope="module")
def data():
    return get_splits()


def test_ks_statistic():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)   # perfect separation
    assert ks_statistic(y, [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)   # no separation


def test_evaluate_known_values():
    y = np.array([0, 0, 0, 1, 1])
    m = evaluate(y, [0.1, 0.2, 0.6, 0.7, 0.4])     # at 0.5: predicted 1 for rows 3 and 4 -> one right, one wrong
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert set(m) == {"roc_auc", "pr_auc", "ks", "precision", "recall", "f1"}
    assert all(0 <= v <= 1 for v in m.values())


def test_four_models_with_right_preprocessing_and_imbalance_handling(data):
    y = data["y_train"]
    models = build_models(y)
    assert list(models) == ["Logistic Regression", "Random Forest", "XGBoost", "XGBoost + SMOTE"]

    # Logistic Regression gets the linear preprocessor (it has the "skewed" log1p branch), trees do not
    assert "skewed" in [t[0] for t in models["Logistic Regression"].named_steps["preprocess"].transformers]
    assert "flagged" in [t[0] for t in models["XGBoost"].named_steps["preprocess"].transformers]

    assert models["Logistic Regression"].named_steps["model"].class_weight == "balanced"
    assert models["Random Forest"].named_steps["model"].class_weight == "balanced_subsample"
    assert models["Random Forest"].named_steps["model"].n_jobs == N_JOBS == 2
    xgb = models["XGBoost"].named_steps["model"]
    assert xgb.scale_pos_weight == pytest.approx((y == 0).sum() / (y == 1).sum())
    assert xgb.n_jobs == 2 and xgb.random_state == 42

    # SMOTE variant: imblearn Pipeline, SMOTE between preprocessing and model, no scale_pos_weight
    smote = models["XGBoost + SMOTE"]
    assert isinstance(smote, ImbPipeline)
    assert [name for name, _ in smote.steps] == ["preprocess", "smote", "model"]
    assert smote.named_steps["model"].scale_pos_weight is None


def test_smote_pipeline_resamples_only_during_fit(data):
    """predict_proba must return one row per input row: SMOTE is skipped at prediction time."""
    X, y = data["X_train"].head(1500), data["y_train"].head(1500)
    pipe = build_models(y)["XGBoost + SMOTE"].fit(X, y)
    proba = pipe.predict_proba(data["X_val"])[:, 1]
    assert len(proba) == len(data["X_val"])
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipelines_handle_missing_values(data):
    X, y = data["X_train"].head(1500), data["y_train"].head(1500)
    X_new = data["X_val"].head(3).copy()
    X_new[["loan_int_rate", "person_emp_length"]] = np.nan
    for name in ["Logistic Regression", "XGBoost"]:
        pipe = build_models(y)[name].fit(X, y)
        assert not np.isnan(pipe.predict_proba(X_new)).any()


def test_perfect_rule_check_counts():
    X = pd.DataFrame({"loan_percent_income": [0.4, 0.5, 0.4, 0.1],
                      "person_home_ownership": ["RENT", "RENT", "OWN", "RENT"]})
    y = pd.Series([1, 1, 0, 0])
    rules = perfect_rule_check(X, y)
    row = rules.loc["loan_percent_income > 0.30 AND renter"]
    assert (row["loans"], row["defaults"], row["default_rate"]) == (2, 2, 1.0)
    assert rules.loc["all loans", "loans"] == 4
