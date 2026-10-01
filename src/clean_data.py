"""
Clean the raw Lending Club files and save them for training.

Input  (data/raw/):        loan_train.csv, loan_test.csv
Output (data/processed/):  train_clean.parquet, test_clean.parquet
                           train_clean.csv,     test_clean.csv   (for viewing in Excel/Numbers)
                           cleaning_report.json                  (summary of what changed)

Run from the project folder:
    python src/clean_data.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "data" / "processed"

TARGET = "loan_status"  # 1 = defaulted, 0 = repaid

# ---------------------------------------------------------------------------
# Columns we remove, and why
# ---------------------------------------------------------------------------

# Known only AFTER the loan was given (payments, recoveries, last payment...).
# Keeping them lets the model "see the future": AUC jumps to ~0.99, which is fake.
LEAKAGE_COLS = [
    "funded_amnt", "funded_amnt_inv",          # decided at approval, ~equal to loan_amnt
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "last_credit_pull_d",
]

# IDs, links and free text: no predictive value or too many unique values.
# `id` is kept separately so predictions can be matched back to a loan.
ID_TEXT_COLS = ["member_id", "url", "desc", "title", "emp_title", "zip_code"]

# More than 90% missing; the same information is already in `pub_rec`.
SPARSE_COLS = ["mths_since_last_record"]

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}
GRADES = "ABCDEFG"


def _to_str(s: pd.Series) -> pd.Series:
    """String version of a column with surrounding spaces removed (works on pandas 2 and 3)."""
    return s.astype("string").str.strip()


def _percent_to_float(s: pd.Series) -> pd.Series:
    """'13.23%' -> 13.23"""
    return pd.to_numeric(_to_str(s).str.rstrip("%"), errors="coerce")


def _month_year_to_date(s: pd.Series) -> pd.Series:
    """'Sep-02' -> 2002-09-01, '(Mar-68)' -> 1968-03-01.

    Two-digit years are ambiguous: pandas reads '68' as 2068. Nothing in this
    dataset is after 2011, so any year after 2011 is moved back 100 years.
    """
    d = pd.to_datetime(_to_str(s), format="%b-%y", errors="coerce")
    future = d.dt.year > 2011
    d = d.where(~future, d - pd.DateOffset(years=100))
    return d


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning to train and test."""
    df = df.copy()

    # 1. Remove leakage, ID/text and very sparse columns
    df = df.drop(columns=[c for c in LEAKAGE_COLS + ID_TEXT_COLS + SPARSE_COLS if c in df.columns])

    # 2. Text -> numbers
    df["term_months"] = pd.to_numeric(_to_str(df.pop("term")).str.extract(r"(\d+)")[0], errors="coerce").astype("Int64")
    df["int_rate"] = _percent_to_float(df["int_rate"])
    df["revol_util"] = _percent_to_float(df["revol_util"])
    df["emp_length"] = _to_str(df["emp_length"]).map(EMP_LENGTH_MAP).astype("Float64")

    # 3. Dates -> credit history length (months), keep issue date for time-based checks and drift
    df["issue_date"] = _month_year_to_date(df.pop("issue_d"))
    earliest = _month_year_to_date(df.pop("earliest_cr_line"))
    df["credit_history_months"] = (
        (df["issue_date"].dt.year - earliest.dt.year) * 12
        + (df["issue_date"].dt.month - earliest.dt.month)
    ).astype("Int64")

    # 4. Tidy categories
    for c in ["grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state"]:
        df[c] = _to_str(df[c])
    df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER"})  # only 3 rows had NONE
    # A1 -> 1, A2 -> 2, ... G5 -> 35 (keeps the order of risk grades)
    df["sub_grade_num"] = (
        df["sub_grade"].str[0].map({g: i for i, g in enumerate(GRADES)}) * 5
        + pd.to_numeric(df["sub_grade"].str[1], errors="coerce")
    ).astype("Int64")

    # 5. Missing value flags (actual filling happens inside the model pipeline, fitted on train only)
    df["ever_delinquent"] = df["mths_since_last_delinq"].notna().astype("int8")

    # 6. Simple, easy-to-explain ratio features
    inc = df["annual_inc"].replace(0, np.nan)
    df["loan_to_income"] = df["loan_amnt"] / inc
    df["installment_to_income"] = (df["installment"] * 12) / inc
    df["revol_bal_to_income"] = df["revol_bal"] / inc

    # 7. Plain column order: id, features, then target / date at the end
    front = ["id"]
    back = [c for c in ["issue_date", TARGET] if c in df.columns]
    middle = [c for c in df.columns if c not in front + back]
    return df[front + middle + back]


def validate(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Checks that fail loudly if something is wrong with the cleaned data."""
    problems = []
    if train["id"].duplicated().any():
        problems.append("duplicate ids in train")
    if not set(train[TARGET].unique()) <= {0, 1}:
        problems.append("target has values other than 0/1")
    leftover = [c for c in LEAKAGE_COLS if c in train.columns]
    if leftover:
        problems.append(f"leakage columns still present: {leftover}")
    if set(train.columns) - {TARGET} != set(test.columns):
        problems.append("train and test columns do not match")
    for name, (lo, hi) in {"int_rate": (0, 40), "dti": (0, 100), "revol_util": (0, 150)}.items():
        v = train[name].dropna()
        if len(v) and (v.min() < lo or v.max() > hi):
            problems.append(f"{name} outside expected range {lo}-{hi}")
    if train["credit_history_months"].dropna().lt(0).any():
        problems.append("negative credit history length (date parsing issue)")
    return problems


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_train = pd.read_csv(RAW_DIR / "loan_train.csv")
    raw_test = pd.read_csv(RAW_DIR / "loan_test.csv")

    train = clean(raw_train)
    test = clean(raw_test)

    problems = validate(train, test)
    if problems:
        raise SystemExit("Cleaning checks failed:\n - " + "\n - ".join(problems))

    # Save: Parquet keeps data types (dates, integers); CSV is for looking at the data
    for name, df in [("train_clean", train), ("test_clean", test)]:
        try:
            df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        except ImportError:
            print("pyarrow not installed, skipping Parquet (pip install pyarrow)")
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    missing = train.isna().mean().round(4)
    report = {
        "raw_train_shape": list(raw_train.shape),
        "raw_test_shape": list(raw_test.shape),
        "clean_train_shape": list(train.shape),
        "clean_test_shape": list(test.shape),
        "default_rate": round(float(train[TARGET].mean()), 4),
        "issue_date_range": [str(train["issue_date"].min().date()), str(train["issue_date"].max().date())],
        "dropped_leakage_columns": LEAKAGE_COLS,
        "dropped_id_text_columns": ID_TEXT_COLS,
        "dropped_sparse_columns": SPARSE_COLS,
        "new_columns": ["term_months", "issue_date", "credit_history_months", "sub_grade_num",
                        "ever_delinquent", "loan_to_income", "installment_to_income", "revol_bal_to_income"],
        "missing_share_after_cleaning": {k: float(v) for k, v in missing[missing > 0].items()},
    }
    (OUT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    print(f"Train: {raw_train.shape} -> {train.shape}")
    print(f"Test:  {raw_test.shape} -> {test.shape}")
    print(f"Default rate: {report['default_rate']:.1%}")
    print(f"Loans issued: {report['issue_date_range'][0]} to {report['issue_date_range'][1]}")
    print("Missing values left (filled later inside the model pipeline):")
    for k, v in report["missing_share_after_cleaning"].items():
        print(f"  {k:24s} {v:.1%}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
