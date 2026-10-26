"""
Quick test report of the model on the TEST split (data the final model has never seen).

    python src/test_model.py

- Logistic Regression and Random Forest: trained here on train + validation (same data as the final model)
- XGBoost: the saved production model (models/loan_default_model.joblib) - nothing is retrained
Prints ROC-AUC for all three, checks the HCL target (ROC-AUC >= 0.85), and shows the classification
report, confusion matrix and APPROVE / REVIEW / REJECT bands for the XGBoost model.
"""

import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from decision import BANDS, assign_bands
from load_data import get_splits
from train import build_models

warnings.filterwarnings("ignore")
PROJECT_DIR = Path(__file__).resolve().parents[1]
TARGET_AUC = 0.85
LINE = "=" * 60


def main() -> None:
    d = get_splits()
    X_fit = pd.concat([d["X_train"], d["X_val"]])
    y_fit = pd.concat([d["y_train"], d["y_val"]])
    X_test, y_test = d["X_test"], d["y_test"]

    print(f"Dataset: Kaggle Credit Risk (cleaned) | {len(X_fit) + len(X_test):,} rows, {X_test.shape[1]} features")
    print(f"Training set: {len(X_fit):,} samples | Testing set: {len(X_test):,} samples (never seen by the final model)")
    print(f"Default rate in training: {y_fit.mean():.2%}")

    models = build_models(y_fit)
    results = {}
    for i, name in enumerate(["Logistic Regression", "Random Forest"], start=1):
        print(f"\n--- Model {i}: {name} {'(Benchmark)' if i == 1 else ''}---")
        pipe = models[name].fit(X_fit, y_fit)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        results[name] = auc
        print(f"{name} ROC-AUC: {auc:.4f} ({auc:.2%})")

    print("\n--- Model 3: XGBoost, tuned + calibrated (Champion / production model) ---")
    model = joblib.load(PROJECT_DIR / "models" / "loan_default_model.joblib")
    meta = json.loads((PROJECT_DIR / "models" / "model_meta.json").read_text())
    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    results["XGBoost"] = auc
    print(LINE)
    print(f"==> XGBoost ROC-AUC Score: {auc:.4f} ({auc:.2%}) <==")
    print(f"==> HCL Target (>= {TARGET_AUC:.0%} AUC): {'PASSED [OK]' if auc >= TARGET_AUC else 'NOT MET'}")
    print(LINE)

    pred = (proba >= 0.5).astype(int)
    print("\nDetailed Performance Report (XGBoost, threshold 0.5):")
    print(classification_report(y_test, pred, target_names=["Fully Paid (0)", "Default (1)"], digits=2))
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print("Confusion matrix:")
    print(f"  Good loans correctly passed : {tn:5d}   | good loans wrongly flagged: {fp}")
    print(f"  Defaults correctly caught   : {tp:5d}   | defaults missed           : {fn}")

    cut = meta["decision_cutoffs"]
    bands = assign_bands(proba, cut["review"], cut["reject"])      # the same band rule as the API
    table = pd.DataFrame({"band": bands, "default": y_test.values}).groupby("band")["default"].agg(["size", "mean"])
    table = table.reindex(BANDS)
    print(f"\nDecision bands (APPROVE < {cut['review']}, REVIEW {cut['review']}-{cut['reject']}, REJECT >= {cut['reject']}):")
    for band, row in table.iterrows():
        print(f"  {band:8s} {int(row['size']):5d} loans ({row['size'] / len(y_test):5.1%}) | real default rate {row['mean']:.1%}")

    print("\nSummary (test ROC-AUC):")
    for name, value in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {name:20s} {value:.4f}")


if __name__ == "__main__":
    main()
