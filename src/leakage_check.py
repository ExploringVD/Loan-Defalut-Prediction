"""
Leakage check + quick baseline models.

1. Single-feature AUC: how well each feature ranks defaulters on its own.
   - numeric feature: the raw value is the score
   - categorical feature: the default rate of its category (learned on the training split) is the score
   The direction (higher = riskier or lower = riskier) is chosen on the training split and the AUC is
   measured on the validation split, so nothing is learned from the data it is scored on.
2. Two quick, untuned models (Pipeline = preprocessor from features.py + model), trained on the training
   split and scored on the validation split: Logistic Regression and Gradient Boosting.
3. Extra checks: validation/test rows that are exact copies of a training row (would inflate the score),
   and the Gradient Boosting AUC when the strongest features are left out (a leak usually sits in one column).

Alarm rules: a single feature above 0.90 AUC, or a quick model above 0.97, is suspicious (the column may
contain the answer) and must be investigated before modelling. Results: reports/leakage_check.md

Run from the project folder:
    python src/leakage_check.py
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from features import build_preprocessor
from load_data import CATEGORICAL_FEATURES, FEATURES, RANDOM_STATE, get_splits

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORT = PROJECT_DIR / "reports" / "leakage_check.md"

SINGLE_FEATURE_ALARM = 0.90
MODEL_ALARM = 0.97


def single_feature_auc(X_train, y_train, X_val, y_val) -> pd.Series:
    """Validation AUC of every feature used alone (see module docstring)."""
    result = {}
    for col in FEATURES:
        if col in CATEGORICAL_FEATURES:
            rates = y_train.groupby(X_train[col]).mean()
            train_score = X_train[col].map(rates)
            val_score = X_val[col].map(rates).fillna(y_train.mean())   # unseen category -> average rate
        else:
            train_score, val_score = X_train[col], X_val[col]
        tr, va = train_score.notna(), val_score.notna()                # rows with a missing value are skipped
        sign = 1 if roc_auc_score(y_train[tr], train_score[tr]) >= 0.5 else -1
        result[col] = roc_auc_score(y_val[va], sign * val_score[va])
    return pd.Series(result).sort_values(ascending=False)


def gradient_boosting(drop_features: list[str] | None = None) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor("tree", drop_features=drop_features)),
        ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


def copies_of_train_rows(X_train, X_other) -> int:
    """How many rows of X_other have exactly the same 11 feature values as some training row."""
    def row_hash(X):
        return pd.util.hash_pandas_object(X[FEATURES].astype(str), index=False)
    return int(row_hash(X_other).isin(set(row_hash(X_train))).sum())


def quick_models(X_train, y_train, X_val, y_val) -> pd.Series:
    models = {
        "Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor("linear")),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "Gradient Boosting": gradient_boosting(),
    }
    result = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        result[name] = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
    return pd.Series(result)


def main() -> None:
    d = get_splits()
    single = single_feature_auc(d["X_train"], d["y_train"], d["X_val"], d["y_val"])
    models = quick_models(d["X_train"], d["y_train"], d["X_val"], d["y_val"])

    copies = {part: copies_of_train_rows(d["X_train"], d[f"X_{part}"]) for part in ["val", "test"]}
    without = {}
    for drop in [[single.index[0]], [single.index[0], single.index[1]], ["loan_percent_income"]]:
        pipe = gradient_boosting(drop).fit(d["X_train"], d["y_train"])
        without[" + ".join(drop)] = roc_auc_score(d["y_val"], pipe.predict_proba(d["X_val"])[:, 1])

    suspicious = single[single > SINGLE_FEATURE_ALARM]
    too_good = models[models > MODEL_ALARM]
    ok = suspicious.empty and too_good.empty

    print("Single-feature validation AUC:")
    for col, auc in single.items():
        kind = "categorical" if col in CATEGORICAL_FEATURES else "numeric"
        print(f"  {col:28s} {auc:.3f}  ({kind})")
    print("\nQuick untuned models, validation AUC:")
    for name, auc in models.items():
        print(f"  {name:28s} {auc:.3f}")
    print(f"\nRows identical to a training row: val {copies['val']}, test {copies['test']}")
    print("Gradient Boosting validation AUC without the strongest features:")
    for dropped, auc in without.items():
        print(f"  without {dropped:36s} {auc:.3f}")
    print(f"\nResult: {'OK - no sign of leakage' if ok else 'CHECK - suspiciously high AUC, investigate'}")

    lines = [
        "# Leakage check and quick baselines",
        "",
        "Made by `python src/leakage_check.py`. Training split fits everything; numbers are on the validation split.",
        f"Alarm rules: single feature > {SINGLE_FEATURE_ALARM:.2f} AUC, or quick model > {MODEL_ALARM:.2f} AUC.",
        "",
        "## Single-feature AUC (validation)",
        "",
        "| Feature | Type | AUC |",
        "|---|---|---|",
        *[f"| {c} | {'categorical' if c in CATEGORICAL_FEATURES else 'numeric'} | {a:.3f} |" for c, a in single.items()],
        "",
        "## Quick untuned models (validation)",
        "",
        "| Model | AUC |",
        "|---|---|",
        *[f"| {n} | {a:.3f} |" for n, a in models.items()],
        "",
        "## Extra checks",
        "",
        f"- Validation rows identical to a training row: {copies['val']}; test rows: {copies['test']}",
        "  (exact duplicates are removed in cleaning, so no loan is in two splits).",
        "- Gradient Boosting validation AUC when the strongest features are left out "
        "(a leak usually sits in one column, so removing it would make the AUC collapse):",
        "",
        "| Left out | AUC |",
        "|---|---|",
        f"| nothing | {models['Gradient Boosting']:.3f} |",
        *[f"| {n} | {a:.3f} |" for n, a in without.items()],
        "",
        f"**Result:** {'no feature or model crosses the alarm lines.' if ok else 'ALARM - investigate before modelling.'}",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"Saved {REPORT.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
