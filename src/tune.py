"""
Tune XGBoost with Optuna, decide on calibration, refit, and evaluate ONCE on the test split.

Steps (the test split is only touched in step 5):
    1. Optuna: 50 trials, objective = mean 5-fold stratified CV ROC-AUC on the TRAINING split.
       MedianPruner stops a trial early when its running CV mean is worse than the median of earlier trials.
       Every trial is logged to MLflow as a nested run.
    2. Fit the best pipeline on the training split and score the VALIDATION split
       (train vs validation AUC gap = overfitting check).
    3. Calibration check on the VALIDATION split: are the predicted probabilities honest?
       (scale_pos_weight makes XGBoost over-predict default.) If the expected calibration error is above
       CALIBRATION_ALARM, wrap the pipeline in CalibratedClassifierCV(isotonic, cv=3) and keep it only if it
       lowers the validation Brier score.
    4. Refit the chosen model on train + validation.
    5. Evaluate ONCE on the test split: ROC-AUC, PR-AUC, KS, Brier score, confusion matrix at 0.5.
    6. Save models/loan_default_model.joblib + models/model_meta.json, register the model in MLflow as
       "loan_default_model" with the alias "production", save charts 15-19 and reports/final_model.md.

Run from the project folder (takes a few minutes):
    python src/tune.py
"""

import json
import os
import platform
import time
from datetime import datetime, timezone

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # silence an MLflow startup hint

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost
from mlflow import MlflowClient
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import build_preprocessor
from load_data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, RANDOM_STATE, TARGET, get_splits
from train import (FIG, INK, LEAKAGE_ALARM, MUTED, N_JOBS, PROJECT_DIR, REPORTS, SKOPS_TRUSTED_TYPES,
                   THRESHOLD, evaluate, setup_mlflow)

MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "loan_default_model.joblib"
META_PATH = MODELS_DIR / "model_meta.json"
REGISTERED_NAME = "loan_default_model"

N_TRIALS = 50
CV_FOLDS = 5
CALIBRATION_ALARM = 0.03   # expected calibration error above 3 percentage points = "poor"
BLUE, ORANGE = "#2a78d6", "#eb6834"
UNTUNED_BASELINES = {"Logistic Regression (quick, untuned)": 0.878, "Gradient Boosting (quick, untuned)": 0.936}


# ---------------------------------------------------------------------------
# Model and search space
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict:
    """The 8 XGBoost hyperparameters Optuna searches."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def build_xgb_pipeline(params: dict, y: pd.Series) -> Pipeline:
    """Tree preprocessor + XGBoost; scale_pos_weight = negatives / positives of the rows it will be fitted on."""
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    model = XGBClassifier(**params, scale_pos_weight=scale_pos_weight, n_jobs=N_JOBS, eval_metric="auc",
                          tree_method="hist", random_state=RANDOM_STATE)
    return Pipeline([("preprocess", build_preprocessor("tree")), ("model", model)])


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Average gap between predicted probability and the real default rate, over 10 equal-width bins
    (weighted by how many loans fall in each bin). 0 = perfectly honest probabilities."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(proba[in_bin].mean() - y_true[in_bin].mean())
    return float(ece)


def overfit_sentence(untuned: pd.Series, train_auc: float, val_auc: float) -> str:
    """Compare the train - validation AUC gap of the untuned (Step 3) and tuned XGBoost."""
    gap_untuned, gap_tuned = untuned["train_auc"] - untuned["val_roc_auc"], train_auc - val_auc
    return (f"Tuning {'reduced' if gap_tuned < gap_untuned else 'did not reduce'} overfitting: the train - validation "
            f"gap went from {gap_untuned:.3f} (untuned) to {gap_tuned:.3f} (tuned), while validation ROC-AUC went "
            f"from {untuned['val_roc_auc']:.3f} to {val_auc:.3f}.")


def full_metrics(y_true, proba) -> dict:
    return {**evaluate(y_true, proba), "brier": brier_score_loss(y_true, proba),
            "ece": expected_calibration_error(y_true, proba)}


# ---------------------------------------------------------------------------
# 1. Optuna
# ---------------------------------------------------------------------------

def make_objective(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold):
    folds = list(cv.split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        start = time.perf_counter()
        with mlflow.start_run(run_name=f"trial_{trial.number:02d}", nested=True):
            mlflow.log_params(params)
            try:
                for step, (tr, va) in enumerate(folds):
                    pipe = build_xgb_pipeline(params, y.iloc[tr]).fit(X.iloc[tr], y.iloc[tr])
                    scores.append(roc_auc_score(y.iloc[va], pipe.predict_proba(X.iloc[va])[:, 1]))
                    trial.report(float(np.mean(scores)), step)        # running CV mean after each fold
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            except optuna.TrialPruned:
                mlflow.set_tag("state", "pruned")
                mlflow.log_metrics({"cv_auc_running_mean": float(np.mean(scores)), "folds_done": len(scores)})
                print(f"  trial {trial.number:2d}  pruned after {len(scores)} folds (running mean {np.mean(scores):.4f})")
                raise
            cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
            mlflow.set_tag("state", "complete")
            mlflow.log_metrics({"cv_auc_mean": cv_mean, "cv_auc_std": cv_std, "seconds": time.perf_counter() - start,
                                **{f"cv_auc_fold_{i + 1}": s for i, s in enumerate(scores)}})
        trial.set_user_attr("cv_auc_std", cv_std)
        print(f"  trial {trial.number:2d}  CV AUC {cv_mean:.4f} +/- {cv_std:.4f}  ({time.perf_counter() - start:.0f}s)")
        return cv_mean

    return objective


def run_study(X: pd.DataFrame, y: pd.Series) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name="xgboost_cv_auc",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),  # judge from fold 2 on
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    study.optimize(make_objective(X, y, cv), n_trials=N_TRIALS)
    return study


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_test_roc(y, p, path):
    fig, ax = plt.subplots(figsize=(6, 5.4))
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="Random guess (AUC 0.500)")
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"Final model (AUC {roc_auc_score(y, p):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (good loans wrongly flagged)",
           ylabel="True positive rate (defaults caught)")
    ax.set_title("ROC curve - test split (used once)")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_test_pr(y, p, path):
    fig, ax = plt.subplots(figsize=(6, 5.4))
    prec, rec, _ = precision_recall_curve(y, p)
    ax.axhline(np.mean(y), color=MUTED, ls=":", lw=1, label=f"Random guess (default rate {np.mean(y):.1%})")
    ax.plot(rec, prec, color=BLUE, lw=2, label=f"Final model (PR-AUC {average_precision_score(y, p):.3f})")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall (share of defaults caught)",
           ylabel="Precision (share of flagged loans that default)")
    ax.set_title("Precision-recall curve - test split")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.4), fontsize=9)
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    total = cm.sum()
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n({v / total:.1%})", ha="center", va="center", fontsize=11,
                color="white" if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["Predicted repaid", "Predicted default"])
    ax.set_yticks([0, 1], ["Actual repaid", "Actual default"])
    ax.set_title(f"Confusion matrix - test split (threshold {THRESHOLD})")
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(val_curves: dict, y_test, p_test, path):
    """Left: the decision (validation split, before/after calibration). Right: the final model on the test split."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
    styles = {"Uncalibrated": (ORANGE, "--", "s"), "Calibrated (isotonic)": (BLUE, "-", "o")}
    for ax in axes:
        ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="Perfectly calibrated")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted probability of default (bin average)")
    for name, (y, p) in val_curves.items():
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        color, ls, marker = styles[name]
        axes[0].plot(mean_pred, frac, color=color, ls=ls, marker=marker, markersize=6, lw=2,
                     label=f"{name} (ECE {expected_calibration_error(y, p):.3f})")
    axes[0].set_ylabel("Actual default rate in the bin")
    axes[0].set_title("Validation split - calibration decision")
    axes[0].legend(loc="upper left", fontsize=9)
    frac, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac, color=BLUE, marker="o", markersize=6, lw=2,
                 label=f"Final model (ECE {expected_calibration_error(y_test, p_test):.3f})")
    axes[1].set_title("Test split - final model")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_optuna_history(study: optuna.Study, path):
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.scatter([t.number for t in complete], [t.value for t in complete], color=BLUE, s=36, zorder=3,
               label=f"Completed trial ({len(complete)})")
    if pruned:
        ax.scatter([t.number for t in pruned], [list(t.intermediate_values.values())[-1] for t in pruned],
                   facecolors="none", edgecolors=MUTED, s=36, zorder=3,
                   label=f"Pruned trial ({len(pruned)}, last running mean)")
    best_so_far = np.maximum.accumulate([t.value for t in complete])
    ax.step([t.number for t in complete], best_so_far, where="post", color=ORANGE, lw=2, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Mean 5-fold CV ROC-AUC (training split)")
    ax.set_title("Optuna search history - XGBoost")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(info: dict, path):
    b, v, c, t = info["best_params"], info["val_tuned"], info["calibration"], info["test"]
    u = info["untuned_xgb"]
    cm = np.array(t["confusion_matrix"])
    params_rows = [f"| {k} | {val:.4g} |" if isinstance(val, float) else f"| {k} | {val} |" for k, val in b.items()]
    lines = [
        "# Final model - tuned XGBoost",
        "",
        f"Made by `python src/tune.py` on {info['training_date'][:10]}. Registered in MLflow as "
        f"`{REGISTERED_NAME}` version {info['version']} with alias `production`; saved as `models/loan_default_model.joblib`.",
        "",
        "## Chosen model",
        "",
        f"- **{info['model_type']}**: tree preprocessor (median imputation + missing flag for loan_int_rate, one-hot "
        "categoricals) + XGBoost with `scale_pos_weight` = negatives / positives"
        + (", wrapped in `CalibratedClassifierCV(method=\"isotonic\", cv=3)`" if c["used"] else "") + ".",
        f"- Tuned with Optuna (TPE sampler, MedianPruner): {info['n_trials']} trials "
        f"({info['n_complete']} completed, {info['n_pruned']} pruned), objective = mean {CV_FOLDS}-fold CV ROC-AUC on the "
        f"training split ({info['n_train']:,} loans).",
        f"- Final fit on train + validation ({info['n_train_val']:,} loans); tested once on {info['n_test']:,} loans.",
        "",
        "### Best hyperparameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
        *params_rows,
        "",
        "## Tuning and overfitting check (validation split, model fitted on the training split)",
        "",
        "| | CV ROC-AUC | Train ROC-AUC | Validation ROC-AUC | Train - validation gap |",
        "|---|---|---|---|---|",
        f"| XGBoost untuned (Step 3) | {u['cv_auc_mean']:.3f} | {u['train_auc']:.3f} | {u['val_roc_auc']:.3f} | "
        f"{u['train_auc'] - u['val_roc_auc']:.3f} |",
        f"| XGBoost tuned | {info['best_cv_auc']:.3f} | {info['train_auc_tuned']:.3f} | {v['roc_auc']:.3f} | "
        f"{info['train_auc_tuned'] - v['roc_auc']:.3f} |",
        "",
        info["gap_sentence"],
        "",
        "## Calibration check (validation split)",
        "",
        "Calibration = can a predicted 30% be read as \"30 of 100 such loans default\"? `scale_pos_weight` pushes the",
        "probabilities up, so we checked. ECE = expected calibration error (average gap between predicted and real",
        f"default rate); \"poor\" means ECE above {CALIBRATION_ALARM}.",
        "",
        "| Version | ROC-AUC | Brier score | ECE |",
        "|---|---|---|---|",
        f"| Uncalibrated | {v['roc_auc']:.3f} | {v['brier']:.4f} | {v['ece']:.3f} |",
        *([f"| Calibrated (isotonic, cv=3) | {c['val']['roc_auc']:.3f} | {c['val']['brier']:.4f} | {c['val']['ece']:.3f} |"]
          if c["val"] else []),
        "",
        f"**Decision:** {c['decision']}",
        "",
        "## Test results (test split used once)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| ROC-AUC | **{t['roc_auc']:.3f}** |",
        f"| PR-AUC | {t['pr_auc']:.3f} |",
        f"| KS statistic | {t['ks']:.3f} |",
        f"| Brier score | {t['brier']:.4f} |",
        f"| Expected calibration error | {t['ece']:.3f} |",
        f"| Precision @ {THRESHOLD} | {t['precision']:.3f} |",
        f"| Recall @ {THRESHOLD} | {t['recall']:.3f} |",
        f"| F1 @ {THRESHOLD} | {t['f1']:.3f} |",
        "",
        f"Confusion matrix at threshold {THRESHOLD} ({info['n_test']:,} test loans):",
        "",
        "| | Predicted repaid | Predicted default |",
        "|---|---|---|",
        f"| **Actual repaid** | {cm[0, 0]:,} (correct) | {cm[0, 1]:,} (good loans wrongly flagged) |",
        f"| **Actual default** | {cm[1, 0]:,} (defaults missed) | {cm[1, 1]:,} (defaults caught) |",
        "",
        "The 0.5 threshold is only a reference point; the real APPROVE / REVIEW / REJECT cut-offs are chosen in Step 5.",
        "",
        "## Compared with the baselines",
        "",
        "| Model | ROC-AUC | Split |",
        "|---|---|---|",
        *[f"| {k} | {val:.3f} | validation |" for k, val in UNTUNED_BASELINES.items()],
        f"| XGBoost untuned (Step 3) | {u['val_roc_auc']:.3f} | validation |",
        f"| XGBoost tuned, fitted on train | {v['roc_auc']:.3f} | validation |",
        f"| **Final model** (train + validation) | **{t['roc_auc']:.3f}** | **test** |",
        "",
        "## Honest note on the 0.85 AUC target",
        "",
        f"The synopsis target of ROC-AUC >= 0.85 is **met without leakage**: the final test ROC-AUC is {t['roc_auc']:.3f}, "
        f"{'in line with' if 0.93 <= t['roc_auc'] <= 0.96 else 'outside'} the ~0.93-0.95 we expected for this dataset "
        f"and below the {LEAKAGE_ALARM} leakage alarm. Every feature is known at application time, all",
        "preprocessing is fitted inside the pipeline on training rows only, and the test split was used once.",
        f"A result above {LEAKAGE_ALARM} would have been treated as a leakage warning (`python src/leakage_check.py`).",
        f"The test score ({t['roc_auc']:.3f}) is a little below the validation score ({v['roc_auc']:.3f}) and close to the CV "
        f"mean ({info['best_cv_auc']:.3f}): normal variation between splits, and the CV mean was the tuning target.",
        "One caveat stays: this dataset contains a perfect rule (every renter whose loan is more than 30% of income",
        "defaulted, 1,650 of 1,650 in the training split - see `reports/model_comparison.md`), so the same model would",
        "probably score lower on real lending data.",
        "",
        "Project history: the first version used Lending Club data, where the honest ROC-AUC was about 0.70 and keeping",
        "post-loan payment columns gave a fake 0.998 (leakage) - see `archive/lending_club/NOTES.md`.",
        "",
        "Charts: `reports/figures/15_test_roc.png`, `16_test_pr.png`, `17_confusion_matrix.png`, `18_calibration.png`,",
        "`19_optuna_history.png`.",
        "",
    ]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start_all = time.perf_counter()
    d = get_splits()
    X_train, y_train, X_val, y_val = d["X_train"], d["y_train"], d["X_val"], d["y_val"]
    untuned = pd.read_csv(REPORTS / "model_comparison.csv").set_index("model").loc["XGBoost"]
    setup_mlflow()
    MODELS_DIR.mkdir(exist_ok=True)

    with mlflow.start_run(run_name="xgboost_tuning") as parent:
        # 1. Optuna on the training split
        print(f"1. Optuna: {N_TRIALS} trials, {CV_FOLDS}-fold CV ROC-AUC on {len(X_train):,} training rows")
        study = run_study(X_train, y_train)
        best = study.best_trial
        best_params = suggest_params(optuna.trial.FixedTrial(best.params))
        n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        print(f"   best trial {best.number}: CV AUC {best.value:.4f} +/- {best.user_attrs['cv_auc_std']:.4f}")

        # 2. Tuned model fitted on train, scored on validation (overfitting check)
        tuned = build_xgb_pipeline(best_params, y_train).fit(X_train, y_train)
        p_val = tuned.predict_proba(X_val)[:, 1]
        train_auc = roc_auc_score(y_train, tuned.predict_proba(X_train)[:, 1])
        val_tuned = full_metrics(y_val, p_val)
        print(f"2. Tuned model: train AUC {train_auc:.4f}, validation AUC {val_tuned['roc_auc']:.4f}. "
              + overfit_sentence(untuned, train_auc, val_tuned["roc_auc"]))

        # 3. Calibration decision on the validation split
        calibration = {"used": False, "val": None}
        val_curves = {"Uncalibrated": (y_val, p_val)}
        if val_tuned["ece"] > CALIBRATION_ALARM:
            calibrated = CalibratedClassifierCV(build_xgb_pipeline(best_params, y_train), method="isotonic", cv=3)
            calibrated.fit(X_train, y_train)
            p_val_cal = calibrated.predict_proba(X_val)[:, 1]
            calibration["val"] = full_metrics(y_val, p_val_cal)
            val_curves["Calibrated (isotonic)"] = (y_val, p_val_cal)
            calibration["used"] = calibration["val"]["brier"] < val_tuned["brier"]
            calibration["decision"] = (
                f"calibration was poor (ECE {val_tuned['ece']:.3f} > {CALIBRATION_ALARM}). Isotonic calibration "
                f"{'lowered' if calibration['used'] else 'did not lower'} the validation Brier score "
                f"({val_tuned['brier']:.4f} -> {calibration['val']['brier']:.4f}) and ECE "
                f"({val_tuned['ece']:.3f} -> {calibration['val']['ece']:.3f}), ROC-AUC {val_tuned['roc_auc']:.3f} -> "
                f"{calibration['val']['roc_auc']:.3f}, so the final model is "
                f"{'the CALIBRATED one' if calibration['used'] else 'the uncalibrated one'}.")
        else:
            calibration["decision"] = (f"calibration is fine (ECE {val_tuned['ece']:.3f} <= {CALIBRATION_ALARM}), "
                                       "so the model is not wrapped.")
        print(f"3. Calibration: {calibration['decision']}")

        # 4. Refit the chosen model on train + validation
        X_train_val, y_train_val = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
        final = build_xgb_pipeline(best_params, y_train_val)
        if calibration["used"]:
            final = CalibratedClassifierCV(final, method="isotonic", cv=3)
        final.fit(X_train_val, y_train_val)
        print(f"4. Final model refitted on train + validation ({len(X_train_val):,} rows)")

        # 5. The ONE evaluation on the test split
        p_test = final.predict_proba(d["X_test"])[:, 1]
        test = full_metrics(d["y_test"], p_test)
        cm = confusion_matrix(d["y_test"], (p_test >= THRESHOLD).astype(int))
        test["confusion_matrix"] = cm.tolist()
        print(f"5. TEST (used once): ROC-AUC {test['roc_auc']:.4f} | PR-AUC {test['pr_auc']:.4f} | KS {test['ks']:.4f} | "
              f"Brier {test['brier']:.4f} | ECE {test['ece']:.4f}")
        if test["roc_auc"] > LEAKAGE_ALARM:
            print(f"   WARNING: test ROC-AUC above {LEAKAGE_ALARM} - check for leakage before using this model.")

        # 6. Charts, MLflow registry, saved files, report
        plot_test_roc(d["y_test"], p_test, FIG / "15_test_roc.png")
        plot_test_pr(d["y_test"], p_test, FIG / "16_test_pr.png")
        plot_confusion(cm, FIG / "17_confusion_matrix.png")
        plot_calibration(val_curves, d["y_test"], p_test, FIG / "18_calibration.png")
        plot_optuna_history(study, FIG / "19_optuna_history.png")

        model_type = "XGBoost pipeline" + (" + isotonic calibration" if calibration["used"] else "")
        mlflow.log_params({f"best__{k}": v for k, v in best_params.items()})
        mlflow.log_params({"model_type": model_type, "calibrated": calibration["used"], "n_trials": N_TRIALS})
        mlflow.log_metrics({"best_cv_auc": best.value, "train_auc_tuned": train_auc,
                            **{f"val_{k}": val for k, val in val_tuned.items()},
                            **{f"test_{k}": val for k, val in test.items() if k != "confusion_matrix"}})
        for name in ["15_test_roc", "16_test_pr", "17_confusion_matrix", "18_calibration", "19_optuna_history"]:
            mlflow.log_artifact(str(FIG / f"{name}.png"))
        logged = mlflow.sklearn.log_model(final, name="model", input_example=X_train.head(5),
                                          registered_model_name=REGISTERED_NAME,
                                          skops_trusted_types=SKOPS_TRUSTED_TYPES)
        version = str(logged.registered_model_version)
        MlflowClient().set_registered_model_alias(REGISTERED_NAME, "production", version)

        info = {
            "model_name": REGISTERED_NAME,
            "version": version,
            "model_type": model_type,
            "calibrated": calibration["used"],
            "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": TARGET,
            "positive_class": "1 = defaulted",
            "features": FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "best_params": best_params,
            "scale_pos_weight": round(float((y_train_val == 0).sum() / (y_train_val == 1).sum()), 4),
            "best_cv_auc": round(best.value, 4),
            "train_auc_tuned": round(train_auc, 4),
            "val_tuned": {k: round(val, 4) for k, val in val_tuned.items()},
            "calibration": {**calibration, "val": None if calibration["val"] is None else
                            {k: round(val, 4) for k, val in calibration["val"].items()}},
            "test": {k: (round(val, 4) if k != "confusion_matrix" else val) for k, val in test.items()},
            "threshold": THRESHOLD,
            "decision_cutoffs": None,   # filled in Step 5
            "n_trials": N_TRIALS, "n_complete": n_complete, "n_pruned": n_pruned,
            "n_train": len(X_train), "n_train_val": len(X_train_val), "n_test": len(d["X_test"]),
            "mlflow_run_id": parent.info.run_id,
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "xgboost": xgboost.__version__, "optuna": optuna.__version__, "mlflow": mlflow.__version__},
        }
        joblib.dump(final, MODEL_PATH)
        META_PATH.write_text(json.dumps(info, indent=2))
        # The report uses the unrounded numbers (rounding twice can change the last digit)
        report_info = {**info, "best_cv_auc": best.value, "train_auc_tuned": train_auc, "val_tuned": val_tuned,
                       "calibration": calibration, "test": test, "untuned_xgb": untuned,
                       "gap_sentence": overfit_sentence(untuned, train_auc, val_tuned["roc_auc"])}
        write_report(report_info, REPORTS / "final_model.md")
        mlflow.log_artifact(str(META_PATH))
        mlflow.log_artifact(str(REPORTS / "final_model.md"))

    print(f"6. Saved {MODEL_PATH.relative_to(PROJECT_DIR)} and {META_PATH.relative_to(PROJECT_DIR)}; "
          f"MLflow '{REGISTERED_NAME}' version {version} -> alias 'production'")
    print(f"   Charts 15-19 and reports/final_model.md written. Total time {(time.perf_counter() - start_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
