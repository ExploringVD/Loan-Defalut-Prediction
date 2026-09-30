"""Checks that the 70/15/15 split has the expected sizes and keeps the default rate."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from load_data import FEATURES, get_splits


def test_split_sizes_and_default_rate():
    data = get_splits()
    assert len(data["X_train"]) == 18_901
    assert len(data["X_val"]) == 4_051
    assert len(data["X_test"]) == 4_051
    for part in ["y_train", "y_val", "y_test"]:
        assert round(data[part].mean(), 3) == 0.147


def test_no_leakage_columns():
    leaky = {"total_pymnt", "recoveries", "out_prncp", "funded_amnt", "last_pymnt_amnt"}
    assert len(FEATURES) == 27
    assert not leaky & set(FEATURES)
    assert "id" not in FEATURES and "issue_date" not in FEATURES
