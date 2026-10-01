"""
Load the cleaned Credit Risk data for model training.

Usage in a notebook or training script:
    from load_data import get_splits, NUMERIC_FEATURES, CATEGORICAL_FEATURES

    data = get_splits()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

Run it directly to see a summary:
    python src/load_data.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

TARGET = "loan_status"
RANDOM_STATE = 42

# The grade is used as a number (loan_grade_num, A=1 ... G=7) and NOT also as the text column
# loan_grade, so the same information is not encoded twice.
NUMERIC_FEATURES = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income", "loan_grade_num",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = ["person_home_ownership", "loan_intent"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_clean() -> pd.DataFrame:
    """Cleaned data made by src/clean_data.py: loan_id, features, loan_grade (text), loan_status."""
    parquet = PROCESSED_DIR / "credit_clean.parquet"
    df = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(PROCESSED_DIR / "credit_clean.csv")
    # Plain numpy types so every scikit-learn / XGBoost version accepts them
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype("float64")
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(object)
    return df


def get_splits(val_size: float = 0.15, test_size: float = 0.15) -> dict:
    """70/15/15 split, stratified so each part keeps the same default rate.

    - train: fit the model
    - val:   tune hyperparameters and the decision threshold
    - test:  touch once at the end to report the final AUC
    """
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size), stratify=y_rest, random_state=RANDOM_STATE
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        # loan_id for every row (same index as X), to trace predictions back to a loan
        "ids": df["loan_id"],
    }


if __name__ == "__main__":
    data = get_splits()
    total = sum(len(data[f"X_{p}"]) for p in ["train", "val", "test"])
    for part in ["train", "val", "test"]:
        X, y = data[f"X_{part}"], data[f"y_{part}"]
        print(f"{part:5s}: {len(X):6,d} rows ({len(X) / total:.0%}) | default rate {y.mean():.1%}")
    print(f"Features: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical = {len(FEATURES)}")
