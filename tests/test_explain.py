"""Tests for the SHAP explanations in src/explain.py (need models/ from python src/tune.py + src/business.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean_data import to_model_input
from explain import (EXAMPLE_APPLICANT, FEATURE_LABELS, META_PATH, MODEL_PATH, explain_one, get_explainer,
                     original_feature, reason_text)
from load_data import FEATURES, get_splits

pytestmark = pytest.mark.skipif(not (MODEL_PATH.exists() and META_PATH.exists()), reason="run src/tune.py first")

RISKY = {"person_age": 23, "person_income": 30000, "person_home_ownership": "RENT", "person_emp_length": 1,
         "loan_intent": "MEDICAL", "loan_grade": "D", "loan_amnt": 12000, "loan_int_rate": 15.5,
         "loan_percent_income": 0.4, "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 3}


def test_labels_for_all_features():
    assert set(FEATURE_LABELS) == set(FEATURES)


def test_original_feature_mapping():
    assert original_feature("loan_int_rate_missing") == "loan_int_rate"
    assert original_feature("person_home_ownership_RENT") == "person_home_ownership"
    assert original_feature("loan_intent_other") == "loan_intent"
    assert original_feature("person_income") == "person_income"
    with pytest.raises(ValueError):
        original_feature("zip_code")


def test_explainer_is_created_once():
    assert get_explainer() is get_explainer()


def test_returns_five_sorted_reasons():
    result = explain_one(RISKY)
    reasons = result["reasons"]
    assert len(reasons) == 5
    assert {"feature", "label", "value", "contribution", "direction"} <= set(reasons[0])
    sizes = [abs(r["contribution"]) for r in reasons]
    assert sizes == sorted(sizes, reverse=True)
    for r in reasons:
        assert r["direction"] == ("increases risk" if r["contribution"] > 0 else "decreases risk")
    assert 0 <= result["probability"] <= 1


def test_contributions_add_up_to_inner_models_log_odds():
    ex = get_explainer()
    X = get_splits()["X_test"].head(300)
    shap_df = ex.shap_by_feature(X)
    np.testing.assert_allclose(ex.base_value + shap_df.sum(axis=1).to_numpy(), ex.log_odds(X), atol=1e-3)
    # the same holds for one applicant through explain_one (all 11 contributions, not only the top 5)
    result = explain_one(RISKY, top_n=len(FEATURES))
    total = result["base_value"] + sum(r["contribution"] for r in result["reasons"])
    assert total == pytest.approx(ex.log_odds(to_model_input(RISKY))[0], abs=1e-2)


def test_decision_band_follows_saved_cutoffs():
    for applicant in [EXAMPLE_APPLICANT, RISKY]:
        r = explain_one(applicant)
        cut = r["cutoffs"]
        expected = "REJECT" if r["probability"] >= cut["reject"] else "REVIEW" if r["probability"] >= cut["review"] else "APPROVE"
        assert r["decision"] == expected
    assert explain_one(EXAMPLE_APPLICANT)["decision"] == "APPROVE"
    assert explain_one(RISKY)["decision"] == "REJECT"


@pytest.mark.parametrize("missing", [{"loan_int_rate": None, "person_emp_length": None}, {"loan_int_rate": np.nan}])
def test_works_with_missing_values(missing):
    result = explain_one({**EXAMPLE_APPLICANT, **missing})
    assert len(result["reasons"]) == 5
    assert not np.isnan(result["probability"])


def test_works_when_optional_fields_are_left_out():
    applicant = {k: v for k, v in EXAMPLE_APPLICANT.items() if k not in ("loan_int_rate", "person_emp_length")}
    assert len(explain_one(applicant)["reasons"]) == 5


def test_reason_text_sentences():
    reasons = [
        {"feature": "loan_percent_income", "value": 0.45, "direction": "increases risk"},
        {"feature": "loan_grade_num", "value": "D", "direction": "increases risk"},
        {"feature": "person_home_ownership", "value": "OWN", "direction": "decreases risk"},
        {"feature": "loan_int_rate", "value": None, "direction": "decreases risk"},
        {"feature": "cb_person_default_on_file", "value": "Y", "direction": "increases risk"},
    ]
    assert reason_text(reasons) == [
        "Loan is 45% of yearly income - increases risk",
        "Grade D loan - increases risk",
        "Owns home - decreases risk",
        "Interest rate not given - decreases risk",
        "Past default on file - increases risk",
    ]
    assert len(reason_text(explain_one(RISKY)["reasons"])) == 5
