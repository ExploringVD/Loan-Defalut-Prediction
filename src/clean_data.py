"""
Clean the raw Kaggle "Credit Risk Dataset" and save it for training.

Input  (data/raw/):        credit_risk_dataset.csv   (download: see README)
Output (data/processed/):  credit_clean.parquet      (used by src/load_data.py)
                           credit_clean.csv          (copy for viewing in Excel/Numbers, not pushed to git)
                           cleaning_report.json      (summary of what changed)

What cleaning does:
    1. removes exact duplicate rows
    2. adds a stable loan_id (= row number in the raw file, so every loan can be traced back)
    3. removes impossible rows (age above 100, more years employed than possible for the age)
    4. adds loan_grade_num (A=1 ... G=7) and turns cb_person_default_on_file Y/N into 1/0
Missing values are NOT filled here: imputation happens inside the model pipeline, fitted on training data only.

Run from the project folder:
    python src/clean_data.py
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "credit_risk_dataset.csv"
OUT_DIR = PROJECT_DIR / "data" / "processed"

TARGET = "loan_status"  # 1 = defaulted, 0 = repaid

RAW_COLUMNS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", TARGET,
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]

GRADE_MAP = {g: i for i, g in enumerate("ABCDEFG", start=1)}  # A=1 (safest) ... G=7 (riskiest)
YES_NO_MAP = {"Y": 1, "N": 0}

MAX_AGE = 100           # nobody older than this takes a loan; the raw data has ages of 123 and 144
MIN_WORKING_AGE = 14    # employment length cannot be longer than (age - 14) years

# Allowed categories (raw data checked) - anything else means the file changed
HOME_OWNERSHIP = {"RENT", "MORTGAGE", "OWN", "OTHER"}
LOAN_INTENT = {"EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"}


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned data and a dict with the number of rows removed per reason."""
    df = raw.copy()
    removed = {}

    # 1. Stable id = row number in the raw file (1-based), set before anything is removed
    df.insert(0, "loan_id", range(1, len(df) + 1))

    # 2. Exact duplicates (all 12 raw columns equal): keep the first copy.
    #    Duplicates could end up in both train and test and make the test score look better than it is.
    dup = df.duplicated(subset=RAW_COLUMNS, keep="first")
    removed["exact_duplicates"] = int(dup.sum())
    df = df[~dup]

    # 3. Impossible values (data entry errors)
    too_old = df["person_age"] > MAX_AGE
    emp_too_long = df["person_emp_length"] > df["person_age"] - MIN_WORKING_AGE  # NaN compares as False -> kept
    removed["age_above_100"] = int(too_old.sum())
    removed["emp_length_above_age_minus_14"] = int((emp_too_long & ~too_old).sum())
    df = df[~(too_old | emp_too_long)]

    # 4. Tidy text and add model-friendly versions
    for c in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        df[c] = df[c].astype("string").str.strip().str.upper()
    df["loan_grade_num"] = df["loan_grade"].map(GRADE_MAP).astype("Int64")
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(YES_NO_MAP).astype("Int64")

    # 5. Column order: id, raw features, new feature, target last
    front = ["loan_id"]
    back = [TARGET]
    middle = [c for c in df.columns if c not in front + back]
    return df[front + middle + back].reset_index(drop=True), removed


# The 11 raw applicant fields (same names as the Kaggle file, without the target). The API, the Streamlit form,
# SHAP explanations and batch scoring all send these and convert them with to_model_input().
APPLICANT_FIELDS = [c for c in RAW_COLUMNS if c != TARGET]
OPTIONAL_FIELDS = ["person_emp_length", "loan_int_rate"]   # may be missing -> imputed inside the model pipeline


def to_model_input(applicants) -> pd.DataFrame:
    """Turn one applicant (dict) or several (list of dicts / DataFrame) with the 11 raw fields into the
    model's input table: loan_grade A-G -> loan_grade_num 1-7, cb_person_default_on_file Y/N -> 1/0,
    columns in the order of load_data.FEATURES. Uses the same maps as clean() so there is only one conversion."""
    from load_data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES   # local import: load_data is the feature list owner

    df = pd.DataFrame([applicants] if isinstance(applicants, dict) else applicants).copy()
    missing = [c for c in APPLICANT_FIELDS if c not in df.columns and c not in OPTIONAL_FIELDS]
    if missing:
        raise ValueError(f"missing applicant fields: {missing}")
    for c in OPTIONAL_FIELDS:
        if c not in df.columns:
            df[c] = None
    for c in ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]:
        df[c] = df[c].astype("string").str.strip().str.upper()
    df["loan_grade_num"] = df["loan_grade"].map(GRADE_MAP)
    df["cb_person_default_on_file"] = df["cb_person_default_on_file"].map(YES_NO_MAP)
    if df["loan_grade_num"].isna().any():
        raise ValueError("loan_grade must be one of A-G")
    if df["cb_person_default_on_file"].isna().any():
        raise ValueError("cb_person_default_on_file must be Y or N")
    X = df[FEATURES].copy()
    X[NUMERIC_FEATURES] = X[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").astype("float64")
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype(object)
    return X


def validate(df: pd.DataFrame) -> list[str]:
    """Checks that fail loudly if something is wrong with the cleaned data."""
    problems = []
    if not set(df[TARGET].unique()) <= {0, 1}:
        problems.append("target has values other than 0/1")
    if df["loan_id"].duplicated().any():
        problems.append("duplicate loan_id")
    if df.duplicated(subset=RAW_COLUMNS).any():
        problems.append("duplicate rows left")
    if df["loan_grade_num"].isna().any():
        problems.append("unknown loan_grade (not A-G)")
    if df["cb_person_default_on_file"].isna().any():
        problems.append("cb_person_default_on_file is not Y/N")
    if not set(df["person_home_ownership"].dropna()) <= HOME_OWNERSHIP:
        problems.append("unexpected person_home_ownership value")
    if not set(df["loan_intent"].dropna()) <= LOAN_INTENT:
        problems.append("unexpected loan_intent value")

    # Sane ranges (NaN is allowed in columns that are imputed later)
    ranges = {
        "person_age": (18, MAX_AGE),
        "person_income": (1, 10_000_000),
        "person_emp_length": (0, MAX_AGE - MIN_WORKING_AGE),
        "loan_amnt": (1, 100_000),
        "loan_int_rate": (1, 40),
        "loan_percent_income": (0, 1),
        "cb_person_cred_hist_length": (0, MAX_AGE),
    }
    for name, (lo, hi) in ranges.items():
        v = df[name].dropna()
        if len(v) and (v.min() < lo or v.max() > hi):
            problems.append(f"{name} outside expected range {lo}-{hi} (found {v.min()}-{v.max()})")
    if (df["person_emp_length"] > df["person_age"] - MIN_WORKING_AGE).any():
        problems.append("person_emp_length longer than possible for the age")

    # Only these two columns may have missing values (they are imputed in the pipeline)
    unexpected_missing = [c for c in df.columns[df.isna().any()] if c not in ("loan_int_rate", "person_emp_length")]
    if unexpected_missing:
        problems.append(f"unexpected missing values in {unexpected_missing}")
    return problems


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(RAW_FILE)
    missing_cols = set(RAW_COLUMNS) - set(raw.columns)
    if missing_cols:
        raise SystemExit(f"{RAW_FILE.name} is missing columns: {sorted(missing_cols)}")

    df, removed = clean(raw)
    problems = validate(df)
    if problems:
        raise SystemExit("Cleaning checks failed:\n - " + "\n - ".join(problems))

    # Parquet keeps data types; CSV is only for looking at the data
    df.to_parquet(OUT_DIR / "credit_clean.parquet", index=False)
    df.to_csv(OUT_DIR / "credit_clean.csv", index=False)

    missing = df.isna().mean()
    report = {
        "raw_file": RAW_FILE.name,
        "raw_shape": list(raw.shape),
        "clean_shape": list(df.shape),
        "rows_removed": removed,
        "rows_removed_total": sum(removed.values()),
        "default_rate_raw": round(float(raw[TARGET].mean()), 4),
        "default_rate_clean": round(float(df[TARGET].mean()), 4),
        "new_columns": ["loan_id", "loan_grade_num"],
        "converted_columns": {"cb_person_default_on_file": "Y/N -> 1/0"},
        "missing_share_after_cleaning": {k: round(float(v), 4) for k, v in missing[missing > 0].items()},
        "note": "Missing values are imputed inside the model pipeline (fitted on the training split only).",
    }
    (OUT_DIR / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    print(f"Raw:   {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    for reason, n in removed.items():
        print(f"  removed {n:4d}  {reason}")
    print(f"Clean: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Default rate: {report['default_rate_clean']:.1%}")
    print("Missing values left (filled later inside the model pipeline):")
    for k, v in missing[missing > 0].items():
        print(f"  {k:22s} {v:.1%}  ({df[k].isna().sum():,} rows)")
    print(f"Saved to {OUT_DIR.relative_to(PROJECT_DIR)}/credit_clean.parquet (+ .csv, cleaning_report.json)")


if __name__ == "__main__":
    main()
