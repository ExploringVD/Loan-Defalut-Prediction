"""
Retraining with a safety check: a new model replaces the production model ONLY if its ROC-AUC on a hold-out
of new labelled loans is not worse.

How it works:
    1. New labelled loans. In real life these are recent loans whose outcome (repaid / defaulted) is now known.
       The dataset has no dates and the simulated drift batch has no true outcomes after its feature changes,
       so for the DEMO the new loans are the same 2,000 intent-resampled test loans as the drift batch, but
       BEFORE the feature changes, with their real outcomes. Half is added to training, the other half is the
       hold-out on which both models are compared.
    2. Candidate = same tuned parameters + same calibration as the production model, trained on the old training
       data (train + validation) plus the new half.
    3. Decision: REPLACE if candidate AUC >= production AUC on the hold-out, else KEEP. Logged to MLflow,
       reports/retrain_log.csv and reports/retrain_report.md.
    4. Only with --apply is a REPLACE carried out: the current model files are backed up to models/backup/<time>/,
       the new model is saved to models/ and registered in MLflow as a new version with the alias "production".

This demonstrates the MECHANISM. The official model metrics stay the Step 4 test numbers. Applying the demo
decision would put test-split loans into the production model, so the Step 4-6 test results would no longer
describe it - that is why the demo does not replace the model unless you ask for it with --apply.

Run from the project folder:
    python src/retrain.py                    # compare + log the decision (production model unchanged)
    python src/retrain.py --apply            # ... and replace the model if the candidate is not worse
    python src/retrain.py --restore models/backup/<folder>   # put a backed-up model back
"""

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from clean_data import to_model_input
from load_data import RANDOM_STATE, TARGET, get_splits
from simulate_batches import intent_resample, raw_test_loans

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"
MODEL_FILE, META_FILE = "loan_default_model.joblib", "model_meta.json"
REPORT_PATH = PROJECT_DIR / "reports" / "retrain_report.md"
LOG_PATH = PROJECT_DIR / "reports" / "retrain_log.csv"
REGISTERED_NAME = "loan_default_model"


def decide(candidate_auc: float, production_auc: float) -> str:
    """The safety rule: never replace the model with a worse one."""
    return "REPLACE" if candidate_auc >= production_auc else "KEEP"


def new_labelled_loans() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Demo 'new loans' (see module docstring) -> (X_add, y_add, X_holdout, y_holdout), split 50/50 stratified."""
    loans = intent_resample(raw_test_loans())                      # same rows as the drift batch, no feature edits
    X, y = to_model_input(loans.drop(columns=["loan_id", TARGET])), loans[TARGET].reset_index(drop=True)
    X_add, X_hold, y_add, y_hold = train_test_split(X, y, test_size=0.5, stratify=y, random_state=RANDOM_STATE)
    return X_add, y_add, X_hold, y_hold


def build_candidate(meta: dict, y: pd.Series):
    from tune import build_xgb_pipeline                          # (imports Optuna/MLflow - only needed here)
    model = build_xgb_pipeline(meta["best_params"], y)
    return CalibratedClassifierCV(model, method="isotonic", cv=3) if meta["calibrated"] else model


def promote(candidate, meta: dict, info: dict, models_dir: Path = MODELS_DIR, register: bool = True) -> dict:
    """Back up the current model files, save the candidate as the new production model, update the meta file
    (and register it in MLflow). Returns the new meta."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = models_dir / "backup" / f"{stamp}_v{meta['version']}"
    backup.mkdir(parents=True)
    for name in (MODEL_FILE, META_FILE):
        shutil.copy2(models_dir / name, backup / name)

    new_meta = {**meta}
    new_meta["version"] = str(int(meta["version"]) + 1)
    if register:
        new_meta["version"] = register_in_mlflow(candidate, info)
    new_meta.update({
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train_val": info["n_train"],
        "retrained_from_version": meta["version"],
        "retrain": {k: info[k] for k in ["candidate_auc", "production_auc", "n_new_added", "n_holdout"]},
        "test_note": f"'test' holds the official Step 4 test metrics of version {meta['version']}; this version "
                     "was retrained and checked on a hold-out of new loans (see 'retrain').",
        "backup_of_previous": str(backup.relative_to(models_dir.parent)),
    })
    joblib.dump(candidate, models_dir / MODEL_FILE)
    (models_dir / META_FILE).write_text(json.dumps(new_meta, indent=2))
    return new_meta


def register_in_mlflow(candidate, info: dict) -> str:
    import mlflow
    from mlflow import MlflowClient

    from train import SKOPS_TRUSTED_TYPES, setup_mlflow
    setup_mlflow()
    with mlflow.start_run(run_name="retrain_promoted"):
        mlflow.log_metrics({"candidate_auc": info["candidate_auc"], "production_auc": info["production_auc"]})
        logged = mlflow.sklearn.log_model(candidate, name="model", registered_model_name=REGISTERED_NAME,
                                          skops_trusted_types=SKOPS_TRUSTED_TYPES)
    version = str(logged.registered_model_version)
    MlflowClient().set_registered_model_alias(REGISTERED_NAME, "production", version)
    return version


def restore(backup: Path, models_dir: Path = MODELS_DIR) -> dict:
    """Put a backed-up model back as production (and move the MLflow alias back to its version)."""
    for name in (MODEL_FILE, META_FILE):
        shutil.copy2(backup / name, models_dir / name)
    meta = json.loads((models_dir / META_FILE).read_text())
    from mlflow import MlflowClient

    from train import setup_mlflow
    setup_mlflow()
    MlflowClient().set_registered_model_alias(REGISTERED_NAME, "production", meta["version"])
    return meta


def log_decision(info: dict) -> None:
    """MLflow run + one line in reports/retrain_log.csv + reports/retrain_report.md."""
    import mlflow

    from train import setup_mlflow
    setup_mlflow()
    with mlflow.start_run(run_name="retrain_check"):
        mlflow.log_metrics({k: info[k] for k in ["candidate_auc", "production_auc", "n_new_added", "n_holdout"]})
        mlflow.set_tags({"decision": info["decision"], "applied": str(info["applied"]),
                         "production_version": info["production_version"]})

    new_file = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(info))
        if new_file:
            writer.writeheader()
        writer.writerow(info)

    gap = info["candidate_auc"] - info["production_auc"]
    REPORT_PATH.write_text("\n".join([
        "# Retraining check",
        "",
        f"Made by `python src/retrain.py` on {info['time'][:10]}. Production model: version {info['production_version']}.",
        "",
        "**This demonstrates the mechanism.** The official model metrics stay the Step 4 test numbers "
        "(`reports/final_model.md`).",
        "",
        "| | |",
        "|---|---|",
        f"| New labelled loans | {info['n_new_added'] + info['n_holdout']:,} test loans, intent-resampled like the drift "
        "batch (50% debt consolidation + medical) but with real features and real outcomes |",
        f"| Added to training | {info['n_new_added']:,} (old training data: {info['n_old']:,} loans -> {info['n_train']:,}) |",
        f"| Hold-out for the comparison | {info['n_holdout']:,} loans ({info['holdout_default_rate']:.1%} defaulted) |",
        f"| Production model ROC-AUC on the hold-out | {info['production_auc']:.4f} |",
        f"| Candidate model ROC-AUC on the hold-out | {info['candidate_auc']:.4f} ({gap:+.4f}) |",
        f"| Rule | replace only if the candidate is not worse |",
        f"| **Decision** | **{info['decision']}** |",
        f"| Carried out? | {'yes - previous model backed up to ' + info['backup'] if info['applied'] else 'no (demo run without --apply)'} |",
        "",
        "Why the demo does not replace the model by default: the new loans come from the test split. A model trained on",
        "them would make the Step 4 test metrics, the business simulation and the SHAP report (all on the test split)",
        "describe a different model. Run `python src/retrain.py --apply` to carry the decision out (a backup is kept;",
        "`python src/retrain.py --restore <backup folder>` puts the old model back).",
        "",
        f"A difference of {abs(gap):.4f} AUC on {info['n_holdout']:,} loans is small - well within normal sampling noise -",
        "so in practice both models are equally good here; the rule simply prefers the newer data when it is not worse.",
        "",
    ]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="replace the production model if the candidate is not worse")
    parser.add_argument("--restore", type=Path, help="put a backed-up model (models/backup/<folder>) back")
    args = parser.parse_args()

    if args.restore:
        meta = restore(args.restore)
        print(f"Restored version {meta['version']} from {args.restore}")
        return

    meta = json.loads((MODELS_DIR / META_FILE).read_text())
    production = joblib.load(MODELS_DIR / MODEL_FILE)
    d = get_splits()
    X_old = pd.concat([d["X_train"], d["X_val"]]).reset_index(drop=True)
    y_old = pd.concat([d["y_train"], d["y_val"]]).reset_index(drop=True)
    X_add, y_add, X_hold, y_hold = new_labelled_loans()
    X_new_train = pd.concat([X_old, X_add], ignore_index=True)
    y_new_train = pd.concat([y_old, y_add], ignore_index=True)

    print(f"Training the candidate on {len(X_new_train):,} loans ({len(X_old):,} old + {len(X_add):,} new) ...")
    candidate = build_candidate(meta, y_new_train).fit(X_new_train, y_new_train)
    production_auc = roc_auc_score(y_hold, production.predict_proba(X_hold)[:, 1])
    candidate_auc = roc_auc_score(y_hold, candidate.predict_proba(X_hold)[:, 1])
    decision = decide(candidate_auc, production_auc)

    info = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "production_version": meta["version"],
        "n_old": len(X_old), "n_new_added": len(X_add), "n_train": len(X_new_train), "n_holdout": len(X_hold),
        "holdout_default_rate": float(y_hold.mean()),
        "production_auc": float(production_auc), "candidate_auc": float(candidate_auc),
        "decision": decision, "applied": False, "backup": "",
    }
    if decision == "REPLACE" and args.apply:
        new_meta = promote(candidate, meta, info)
        info.update(applied=True, backup=new_meta["backup_of_previous"])
        print(f"Model replaced: version {meta['version']} -> {new_meta['version']} "
              f"(backup in {new_meta['backup_of_previous']}). Restart the API to load it.")
    log_decision(info)
    print(f"Hold-out ROC-AUC: production {production_auc:.4f} | candidate {candidate_auc:.4f} -> {decision}"
          f"{'' if info['applied'] or decision == 'KEEP' else ' (not applied: run with --apply)'}")
    print(f"Logged to MLflow, {LOG_PATH.relative_to(PROJECT_DIR)} and {REPORT_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
