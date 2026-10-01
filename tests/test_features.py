"""Tests for the preprocessing in src/features.py."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import LINEAR_DROP_FEATURES, build_preprocessor, get_feature_names
from load_data import get_splits

KINDS = ["linear", "tree"]


@pytest.fixture(scope="module")
def data():
    return get_splits()


@pytest.mark.parametrize("kind", KINDS)
def test_no_nans_and_same_columns_for_all_splits(kind, data):
    pre = build_preprocessor(kind).fit(data["X_train"])
    outputs = [pre.transform(data[part]) for part in ["X_train", "X_val", "X_test"]]
    for out in outputs:
        assert not np.isnan(out).any()
        assert out.shape[1] == outputs[0].shape[1]


@pytest.mark.parametrize("kind", KINDS)
def test_unseen_and_missing_categories_do_not_crash(kind, data):
    pre = build_preprocessor(kind).fit(data["X_train"])
    X_new = data["X_val"].head(3).copy()
    X_new["loan_intent"] = "SPACE_TRAVEL"                 # intent never seen in training
    X_new["person_home_ownership"] = "BOAT"               # home ownership never seen in training
    X_new.loc[X_new.index[0], "person_home_ownership"] = None
    out = pre.transform(X_new)
    assert out.shape == (3, len(get_feature_names(pre)))
    assert not np.isnan(out).any()
    # unknown category -> all its one-hot columns are 0
    names = get_feature_names(pre)
    intent_cols = [i for i, n in enumerate(names) if n.startswith("loan_intent_")]
    assert (out[:, intent_cols] == 0).all()


@pytest.mark.parametrize("kind", KINDS)
def test_returned_unfitted(kind):
    with pytest.raises(NotFittedError):
        check_is_fitted(build_preprocessor(kind))


def test_fit_uses_only_given_data(data):
    """Medians, scaler means and categories must come from the fitted rows only."""
    X_small = data["X_train"].head(500)
    pre = build_preprocessor("linear").fit(X_small)

    num_pipe = pre.named_transformers_["num"]
    num_cols = list(pre.transformers_[1][2])
    expected_median = X_small[num_cols].median().to_numpy()
    np.testing.assert_allclose(num_pipe.named_steps["impute"].statistics_, expected_median)

    skewed_pipe = pre.named_transformers_["skewed"]
    skewed_cols = list(pre.transformers_[0][2])
    expected_mean = np.log1p(X_small[skewed_cols]).mean().to_numpy()
    np.testing.assert_allclose(skewed_pipe.named_steps["scale"].mean_[: len(skewed_cols)], expected_mean)

    # fitting again on the same rows gives the same output, whatever other data exists
    again = build_preprocessor("linear").fit(X_small)
    np.testing.assert_allclose(pre.transform(data["X_val"]), again.transform(data["X_val"]))


def test_tree_imputer_median_from_train(data):
    pre = build_preprocessor("tree").fit(data["X_train"])
    flagged = pre.named_transformers_["flagged"]
    assert flagged.statistics_[0] == data["X_train"]["loan_int_rate"].median()


def test_feature_names_are_readable(data):
    lin = build_preprocessor("linear").fit(data["X_train"])
    names = get_feature_names(lin)
    assert len(names) == lin.transform(data["X_val"].head(2)).shape[1]
    assert len(set(names)) == len(names)
    assert not any("__" in n or "missingindicator" in n or "infrequent_sklearn" in n for n in names)
    assert "person_emp_length_missing" in names
    assert "loan_intent_MEDICAL" in names
    for dropped in LINEAR_DROP_FEATURES:
        assert dropped not in names
    assert "loan_grade_num" in names and "person_age" in names      # kept side of each correlated pair

    tree = build_preprocessor("tree").fit(data["X_train"])
    tree_names = get_feature_names(tree)
    assert "loan_int_rate" in tree_names and "cb_person_cred_hist_length" in tree_names
    assert "loan_int_rate_missing" in tree_names
    assert "person_emp_length_missing" not in tree_names


def test_rare_categories_grouped_as_other(data):
    """OTHER home ownership is rare: in 2,000 training rows it appears fewer than 20 times -> "_other"."""
    X_small = data["X_train"].head(2000)
    assert (X_small["person_home_ownership"] == "OTHER").sum() < 20
    names = get_feature_names(build_preprocessor("linear").fit(X_small))
    assert "person_home_ownership_other" in names
    assert "person_home_ownership_OTHER" not in names


def test_drop_list_is_configurable(data):
    pre = build_preprocessor("linear", drop_features=["person_income", "loan_intent"]).fit(data["X_train"])
    names = get_feature_names(pre)
    assert "person_income" not in names
    assert not any(n.startswith("loan_intent_") for n in names)
    assert "loan_int_rate" in names      # default drop list replaced, not added to


def test_get_feature_names_accepts_pipeline(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([("preprocess", build_preprocessor("linear")), ("model", LogisticRegression(max_iter=500))])
    pipe.fit(data["X_train"], data["y_train"])
    assert len(get_feature_names(pipe)) == pipe.named_steps["model"].coef_.shape[1]


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_preprocessor("neural")
