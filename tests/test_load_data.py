"""Checks that the 70/15/15 split has the expected sizes and keeps the default rate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from load_data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, get_splits


def test_split_sizes_and_default_rate():
    data = get_splits()
    assert len(data["X_train"]) == 22_685
    assert len(data["X_val"]) == 4_862
    assert len(data["X_test"]) == 4_862
    for part in ["y_train", "y_val", "y_test"]:
        assert round(data[part].mean(), 3) in (0.218, 0.219)


def test_splits_do_not_overlap():
    data = get_splits()
    train, val, test = (set(data[f"X_{p}"].index) for p in ["train", "val", "test"])
    assert not (train & val) and not (train & test) and not (val & test)


def test_feature_lists():
    assert len(NUMERIC_FEATURES) == 9 and len(CATEGORICAL_FEATURES) == 2
    assert len(FEATURES) == 11
    # grade is used once (as a number), ids and the target are never features
    assert "loan_grade_num" in FEATURES and "loan_grade" not in FEATURES
    assert "loan_id" not in FEATURES and "loan_status" not in FEATURES
    assert "issue_date" not in FEATURES


def test_feature_types():
    X = get_splits()["X_train"]
    assert list(X.columns) == FEATURES
    assert all(X[c].dtype == "float64" for c in NUMERIC_FEATURES)
    assert set(X["cb_person_default_on_file"].unique()) == {0.0, 1.0}
