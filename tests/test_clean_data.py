"""Tests for the cleaning rules in src/clean_data.py, on a small hand-made table."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clean_data import clean, validate


def raw_rows() -> pd.DataFrame:
    base = dict(person_age=30, person_income=50_000, person_home_ownership="RENT", person_emp_length=5.0,
                loan_intent="MEDICAL", loan_grade="B", loan_amnt=10_000, loan_int_rate=11.0, loan_status=0,
                loan_percent_income=0.2, cb_person_default_on_file="N", cb_person_cred_hist_length=4)
    rows = [
        base,                                                         # 1 normal
        base,                                                         # 2 exact duplicate of 1 -> removed
        {**base, "person_age": 144},                                  # 3 impossible age -> removed
        {**base, "person_age": 22, "person_emp_length": 123.0},       # 4 employed longer than possible -> removed
        {**base, "loan_grade": "G", "cb_person_default_on_file": "Y", "loan_status": 1},   # 5
        {**base, "loan_int_rate": np.nan, "person_emp_length": np.nan, "loan_amnt": 5_000},  # 6 missing values
    ]
    return pd.DataFrame(rows)


def test_duplicates_and_impossible_rows_removed():
    df, removed = clean(raw_rows())
    assert removed == {"exact_duplicates": 1, "age_above_100": 1, "emp_length_above_age_minus_14": 1}
    assert list(df["loan_id"]) == [1, 5, 6]          # loan_id = row number in the raw file


def test_new_and_converted_columns():
    df, _ = clean(raw_rows())
    assert list(df["loan_grade_num"]) == [2, 7, 2]
    assert list(df["cb_person_default_on_file"]) == [0, 1, 0]
    assert df.columns[0] == "loan_id" and df.columns[-1] == "loan_status"


def test_missing_values_are_not_imputed():
    df, _ = clean(raw_rows())
    assert df["loan_int_rate"].isna().sum() == 1
    assert df["person_emp_length"].isna().sum() == 1


def test_validate_passes_on_clean_data_and_fails_loudly():
    df, _ = clean(raw_rows())
    assert validate(df) == []

    bad = df.copy()
    bad.loc[0, "loan_status"] = 2
    bad.loc[1, "loan_int_rate"] = 99.0
    problems = validate(bad)
    assert any("target" in p for p in problems)
    assert any("loan_int_rate" in p for p in problems)

    dup = pd.concat([df, df.iloc[[0]].assign(loan_id=99)], ignore_index=True)
    assert any("duplicate rows" in p for p in validate(dup))
