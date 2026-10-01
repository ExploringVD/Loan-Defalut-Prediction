"""Tests for the decision bands and the NPA simulation in src/business.py."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from business import (MAX_GOOD_REJECTED, MIN_DEFAULTERS_FLAGGED, assign_bands, band_table, choose_cutoffs,
                      interest_rate_filled, simulate, tradeoff_table)
from tune import META_PATH


def test_assign_bands_boundaries():
    bands = assign_bands([0.0, 0.0999, 0.10, 0.3499, 0.35, 1.0], review_cutoff=0.10, reject_cutoff=0.35)
    assert list(bands) == ["APPROVE", "APPROVE", "REVIEW", "REVIEW", "REJECT", "REJECT"]


def test_choose_cutoffs_follows_the_two_rules():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 5000)
    p = np.clip(np.where(y == 1, rng.normal(0.6, 0.25, 5000), rng.normal(0.15, 0.1, 5000)), 0, 1)
    review, reject = choose_cutoffs(y, p)
    t = tradeoff_table(y, p).set_index("cutoff")
    assert review < reject
    assert t.loc[reject, "good_loans_lost"] <= MAX_GOOD_REJECTED
    assert t.loc[review, "defaulters_caught"] >= MIN_DEFAULTERS_FLAGGED
    # reject is the LOWEST cut-off meeting its rule, review the HIGHEST meeting its rule
    assert (t.loc[t.index < reject, "good_loans_lost"] > MAX_GOOD_REJECTED).all()
    assert (t.loc[t.index > review, "defaulters_caught"] < MIN_DEFAULTERS_FLAGGED).all()


def test_choose_cutoffs_refuses_overlapping_bands():
    """If the 90%-of-defaulters cut-off is above the 2%-of-good-loans cut-off, there is no REVIEW band -> error."""
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, 5000)
    p = np.clip(np.where(y == 1, rng.normal(0.7, 0.2, 5000), rng.normal(0.15, 0.1, 5000)), 0, 1)
    with pytest.raises(ValueError):
        choose_cutoffs(y, p)


def test_band_table_shares_add_up():
    y = np.array([0, 0, 1, 1, 0, 1])
    t = band_table(y, ["APPROVE", "APPROVE", "REVIEW", "REJECT", "REJECT", "REJECT"]).set_index("band")
    assert t["loans"].sum() == 6
    assert t["share_of_defaulters"].sum() == pytest.approx(1.0)
    assert t["share_of_good_loans"].sum() == pytest.approx(1.0)
    assert t.loc["REJECT", "default_rate"] == pytest.approx(2 / 3)


def test_simulate_counts_and_money():
    y = np.array([0, 0, 1, 1])
    bands = np.array(["APPROVE", "REVIEW", "REVIEW", "REJECT"])
    sim = simulate(y, bands, loan_amnt=[1000, 2000, 3000, 4000], int_rate=[10, 10, 10, 10]).set_index("policy")
    base, a, b = sim.iloc[0], sim.iloc[1], sim.iloc[2]
    assert (base["defaults_approved"], a["defaults_approved"], b["defaults_approved"]) == (2, 1, 0)
    assert base["default_amount_approved"] == 7000 and a["default_amount_approved"] == 3000
    assert a["npa_reduction_amount"] == pytest.approx(1 - 3000 / 7000)
    assert b["npa_reduction_count"] == pytest.approx(1.0)
    assert (a["good_loans_lost"], b["good_loans_lost"]) == (0, 1)
    assert base["net_result"] == pytest.approx(0.1 * 3000 - 7000)     # interest on 2 good loans - 2 defaults
    assert base["npa_reduction_count"] == 0


def test_missing_interest_rate_uses_grade_median_from_training():
    X_train = pd.DataFrame({"loan_grade_num": [1, 1, 2], "loan_int_rate": [7.0, 9.0, 12.0]})
    X = pd.DataFrame({"loan_grade_num": [1, 2], "loan_int_rate": [np.nan, 11.0]})
    assert list(interest_rate_filled(X, X_train)) == [8.0, 11.0]


@pytest.mark.skipif(not META_PATH.exists(), reason="run python src/tune.py and src/business.py first")
def test_cutoffs_saved_in_meta():
    cut = json.loads(META_PATH.read_text()).get("decision_cutoffs")
    if cut is None:
        pytest.skip("run python src/business.py first")
    assert 0 < cut["review"] < cut["reject"] < 1
    assert cut["chosen_on"].startswith("validation")
