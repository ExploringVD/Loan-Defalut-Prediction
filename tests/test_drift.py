"""Tests for drift monitoring (src/drift.py), the batch simulator (src/simulate_batches.py) and the
retraining rule (src/retrain.py)."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from conftest import needs_model
from drift import DRIFT_SHARE_ALARM, PSI_ALARM, column_drifted, drift_status, psi
from retrain import decide, promote
from simulate_batches import (BATCH_SIZE, INCOME_FACTOR, RATE_SHIFT, SHIFTED_INTENTS, make_batches, raw_test_loans)

# ---------------------------------------------------------------------------
# PSI and the alert rule
# ---------------------------------------------------------------------------


def test_psi_same_distribution_is_about_zero():
    rng = np.random.default_rng(42)
    assert psi(rng.normal(0, 1, 20_000), rng.normal(0, 1, 5_000)) < 0.01
    assert psi([0.1, 0.2, 0.3] * 100, [0.1, 0.2, 0.3] * 100) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_shift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 20_000)
    small, large = psi(ref, rng.normal(0.1, 1, 5_000)), psi(ref, rng.normal(1.0, 1, 5_000))
    assert small < 0.1 < PSI_ALARM < large


def test_psi_handles_many_tied_values():
    """Calibrated probabilities have many identical values: duplicate bin edges must not crash."""
    ref = np.r_[np.zeros(800), np.full(150, 0.5), np.ones(50)]
    cur = np.r_[np.zeros(500), np.full(250, 0.5), np.ones(250)]
    assert np.isfinite(psi(ref, cur)) and psi(ref, cur) > PSI_ALARM


@pytest.mark.parametrize("share,psi_value,status", [
    (0.0, 0.0, "NO DRIFT"),
    (DRIFT_SHARE_ALARM, PSI_ALARM, "NO DRIFT"),            # exactly on the line is not above it
    (4 / 11, 0.05, "DRIFT DETECTED"),                       # too many features drifted
    (0.0, 0.25, "DRIFT DETECTED"),                          # predictions moved too much
])
def test_alert_rule(share, psi_value, status):
    assert drift_status(share, psi_value) == status


def test_column_drift_direction_depends_on_the_test():
    assert column_drifted(0.15, "Wasserstein distance (normed)", 0.1)        # distance: big = drift
    assert not column_drifted(0.05, "Jensen-Shannon distance", 0.1)
    assert column_drifted(0.01, "K-S p_value", 0.05)                         # p-value: small = drift
    assert not column_drifted(0.40, "chi-square p_value", 0.05)


# ---------------------------------------------------------------------------
# Batch simulator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def batches():
    return make_batches()


@pytest.fixture(scope="module")
def test_loans():
    return raw_test_loans().set_index("loan_id")


def test_batch_sizes_and_columns(batches):
    for batch in batches.values():
        assert len(batch) == BATCH_SIZE and batch["loan_id"].is_unique
        assert "loan_status" not in batch.columns


def test_no_drift_batch_is_unchanged(batches, test_loans):
    nd = batches["no_drift"].set_index("loan_id")
    pd.testing.assert_frame_equal(nd, test_loans.loc[nd.index, nd.columns], check_dtype=False)


def test_drift_batch_shifts(batches, test_loans):
    dr = batches["drift"].set_index("loan_id")
    original = test_loans.loc[dr.index]
    rate_known = original["loan_int_rate"].notna()
    np.testing.assert_allclose(dr.loc[rate_known, "loan_int_rate"], original.loc[rate_known, "loan_int_rate"] + RATE_SHIFT)
    assert dr.loc[~rate_known, "loan_int_rate"].isna().all()                  # missing stays missing
    np.testing.assert_allclose(dr["person_income"], (original["person_income"] * INCOME_FACTOR).round())
    np.testing.assert_allclose(dr["loan_percent_income"], (dr["loan_amnt"] / dr["person_income"]).round(2))
    assert dr["loan_intent"].isin(SHIFTED_INTENTS).mean() == pytest.approx(0.50)
    assert test_loans["loan_intent"].isin(SHIFTED_INTENTS).mean() == pytest.approx(0.35, abs=0.02)
    unchanged = ["person_age", "person_home_ownership", "loan_grade", "loan_amnt", "cb_person_default_on_file"]
    pd.testing.assert_frame_equal(dr[unchanged], original[unchanged], check_dtype=False)


def test_batches_are_reproducible_and_valid_for_the_api(batches):
    again = make_batches()
    for name in batches:
        pd.testing.assert_frame_equal(batches[name], again[name])
    from app.schemas import ApplicantIn
    for row in batches["drift"].head(200).to_dict("records"):
        row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items() if k != "loan_id"}
        ApplicantIn(applicant_name="batch", **row)                            # raises if the API would reject it


# ---------------------------------------------------------------------------
# The real drift check (needs the saved model)
# ---------------------------------------------------------------------------

@needs_model
def test_drift_check_flags_the_drift_batch_only(batches):
    from drift import run_drift_check
    quiet = run_drift_check(batches["no_drift"], "no_drift", save=False)
    alarm = run_drift_check(batches["drift"], "drift", save=False)
    assert quiet["status"] == "NO DRIFT" and quiet["n_drifted"] == 0
    assert alarm["status"] == "DRIFT DETECTED"
    assert {"loan_int_rate", "person_income", "loan_percent_income", "loan_intent"} <= set(alarm["drifted_features"])
    assert alarm["mean_probability"]["current"] > alarm["mean_probability"]["reference"] + 0.05
    assert alarm["prediction_psi"] > quiet["prediction_psi"]


# ---------------------------------------------------------------------------
# Retraining rule and model replacement
# ---------------------------------------------------------------------------

def test_retrain_rule_never_accepts_a_worse_model():
    assert decide(0.951, 0.950) == "REPLACE"
    assert decide(0.950, 0.950) == "REPLACE"          # not worse
    assert decide(0.949, 0.950) == "KEEP"


def test_promote_keeps_a_backup(tmp_path):
    (tmp_path / "loan_default_model.joblib").write_bytes(b"old model")
    meta = {"version": "1", "best_params": {}, "test": {"roc_auc": 0.95}}
    (tmp_path / "model_meta.json").write_text(json.dumps(meta))
    info = {"candidate_auc": 0.96, "production_auc": 0.95, "n_new_added": 10, "n_holdout": 10, "n_train": 100}

    new_meta = promote({"new": "model"}, meta, info, models_dir=tmp_path, register=False)

    backups = list((tmp_path / "backup").iterdir())
    assert len(backups) == 1 and (backups[0] / "loan_default_model.joblib").read_bytes() == b"old model"
    assert json.loads((backups[0] / "model_meta.json").read_text())["version"] == "1"
    assert joblib.load(tmp_path / "loan_default_model.joblib") == {"new": "model"}
    saved = json.loads((tmp_path / "model_meta.json").read_text())
    assert saved == new_meta and saved["version"] == "2" and saved["retrained_from_version"] == "1"
    assert saved["test"] == meta["test"]               # official Step 4 metrics are kept, with a note
