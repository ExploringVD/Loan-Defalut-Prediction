"""
Step 2: load the cleaned data for model training.

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

NUMERIC_FEATURES = [
    "loan_amnt", "term_months", "int_rate", "installment", "sub_grade_num",
    "emp_length", "annual_inc", "dti",
    "delinq_2yrs", "inq_last_6mths", "mths_since_last_delinq", "ever_delinquent",
    "open_acc", "total_acc", "pub_rec", "pub_rec_bankruptcies",
    "revol_bal", "revol_util", "credit_history_months",
    "loan_to_income", "installment_to_income", "revol_bal_to_income",
]
CATEGORICAL_FEATURES = [
    "grade", "home_ownership", "verification_status", "purpose", "addr_state",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _read(name: str) -> pd.DataFrame:
    parquet = PROCESSED_DIR / f"{name}.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        df = pd.read_csv(PROCESSED_DIR / f"{name}.csv", parse_dates=["issue_date"])
    # Plain numpy types so every scikit-learn / XGBoost version accepts them
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype("float64")
    df[CATEGORICAL_FEATURES] = df[CATEGORICAL_FEATURES].astype(object)
    return df


def load_clean_train() -> pd.DataFrame:
    """Cleaned training file: id, features, issue_date, loan_status."""
    return _read("train_clean")


def load_clean_test() -> pd.DataFrame:
    """Cleaned Kaggle test file (no loan_status). Only used for the Kaggle submission."""
    return _read("test_clean")


def get_splits(val_size: float = 0.15, test_size: float = 0.15) -> dict:
    """70/15/15 split of the labelled data, stratified so each part keeps the same default rate.

    - train: fit the model
    - val:   tune hyperparameters and the decision threshold
    - test:  touch once at the end to report the final AUC
    """
    df = load_clean_train()
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
        # kept aside for drift monitoring demos (not a model feature)
        "issue_date": df.loc[X.index, "issue_date"],
        "ids": df.loc[X.index, "id"],
    }


if __name__ == "__main__":
    data = get_splits()
    for part in ["train", "val", "test"]:
        X, y = data[f"X_{part}"], data[f"y_{part}"]
        print(f"{part:5s}: {len(X):6,d} rows | default rate {y.mean():.1%}")
    print(f"Features: {len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical = {len(FEATURES)}")
    print(f"Kaggle test file: {len(load_clean_test()):,} rows (no labels)")
