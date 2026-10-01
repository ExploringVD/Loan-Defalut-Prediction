"""
Train and compare four models on the training split, each as one full Pipeline (preprocessing + model).

    1. Logistic Regression   linear preprocessor, class_weight="balanced"
    2. Random Forest         tree preprocessor,   class_weight="balanced_subsample"
    3. XGBoost               tree preprocessor,   scale_pos_weight = negatives / positives
    4. XGBoost + SMOTE       tree preprocessor,   SMOTE oversampling inside an imblearn Pipeline

For each model:
    - 5-fold stratified cross-validation ROC-AUC on the training split (mean and std)
    - fit on the whole training split, evaluate on the validation split:
      ROC-AUC, PR-AUC, KS statistic, precision / recall / F1 at threshold 0.5
    - log params, metrics, the fitted pipeline and plots to MLflow (local: sqlite:///mlflow.db, artifacts in mlruns/)

The test split is NOT used here (it is used once, in Step 4).

Outputs:
    reports/model_comparison.csv, reports/model_comparison.md
    reports/figures/13_roc_curves_validation.png, reports/figures/14_pr_curves_validation.png

Run from the project folder:
    python src/train.py
Then look at the runs:
    mlflow ui --backend-store-uri sqlite:///mlflow.db      (open http://127.0.0.1:5000)
"""

import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # silence an MLflow startup hint

import matplotlib
matplotlib.use("Agg")                                      # save charts to files, no window
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import build_preprocessor
from leakage_check import perfect_rule_check
from load_data import FEATURES, RANDOM_STATE, get_splits

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_DIR / "reports"
FIG = REPORTS / "figures"

TRACKING_URI = "sqlite:///mlflow.db"                       # relative to the project folder (scripts run from there)
ARTIFACT_DIR = PROJECT_DIR / "mlruns"
EXPERIMENT = "loan-default"

# MLflow saves sklearn models with skops (safer than pickle: it only loads types it is told to trust).
# These are the non-standard types inside our pipelines (Step 3 models + the calibrated model of Step 4),
# all from libraries we install ourselves.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype", "sklearn.tree._tree.Tree", "sklearn.calibration._CalibratedClassifier",
    "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier",
    "imblearn.pipeline.Pipeline", "imblearn.over_sampling._smote.base.SMOTE",
]
# MLflow hint about the model OUTPUT (predicted class 0/1 is an integer) - not a problem here
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer column")

N_JOBS = 2               # 8 GB RAM laptop: never -1
CV_FOLDS = 5
THRESHOLD = 0.5
LEAKAGE_ALARM = 0.97     # validation AUC above this -> stop and check for leakage
BASELINES = {"Logistic Regression (untuned, no class weights)": 0.878, "Gradient Boosting (untuned)": 0.936}

# Chart style (same as the EDA notebook). Line colours passed the colour-blind check;
# the SMOTE variant is also dashed so it does not depend on colour alone.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
STYLE = {
    "Logistic Regression": ("#2a78d6", "-"),
    "Random Forest": ("#eb6834", "-"),
    "XGBoost": ("#1baf7a", "-"),
    "XGBoost + SMOTE": ("#eda100", "--"),
}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_models(y_train: pd.Series) -> dict:
    """The four UNFITTED pipelines. Every preprocessing step is inside the pipeline (fitted on training folds only)."""
    neg_pos_ratio = float((y_train == 0).sum() / (y_train == 1).sum())   # about 3.6 here
    xgb_common = dict(n_jobs=N_JOBS, eval_metric="auc", tree_method="hist", random_state=RANDOM_STATE)
    return {
        "Logistic Regression": Pipeline([
            ("preprocess", build_preprocessor("linear")),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", build_preprocessor("tree")),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                             n_jobs=N_JOBS, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("preprocess", build_preprocessor("tree")),
            ("model", XGBClassifier(scale_pos_weight=neg_pos_ratio, **xgb_common)),
        ]),
        # imblearn Pipeline: SMOTE runs only during fit (on the training rows), never during predict
        "XGBoost + SMOTE": ImbPipeline([
            ("preprocess", build_preprocessor("tree")),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", XGBClassifier(**xgb_common)),
        ]),
    }


IMBALANCE_HANDLING = {
    "Logistic Regression": 'class_weight="balanced"',
    "Random Forest": 'class_weight="balanced_subsample"',
    "XGBoost": "scale_pos_weight = negatives / positives",
    "XGBoost + SMOTE": "SMOTE oversampling (training rows only)",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ks_statistic(y_true, proba) -> float:
    """Kolmogorov-Smirnov: the biggest gap between the score distributions of defaulters and non-defaulters
    (= max of TPR - FPR over all thresholds). 0 = no separation, 1 = perfect. Banks often quote it."""
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def evaluate(y_true, proba, threshold: float = THRESHOLD) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),     # area under the precision-recall curve
        "ks": ks_statistic(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def model_params(pipe) -> dict:
    """Hyperparameters of the model step, only simple values (for MLflow)."""
    params = pipe.named_steps["model"].get_params()
    return {k: v for k, v in params.items() if v is not None and isinstance(v, (int, float, str, bool))}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def plot_roc(curves: dict, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="Random guess (AUC 0.500)")
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        color, ls = STYLE[name]
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate (good loans wrongly flagged)")
    ax.set_ylabel("True positive rate (defaults caught)")
    ax.set_title("ROC curves - validation split")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(path)
    return fig


def plot_pr(curves: dict, path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    base_rate = float(np.mean(next(iter(curves.values()))[0]))
    ax.axhline(base_rate, color=MUTED, ls=":", lw=1, label=f"Random guess (default rate {base_rate:.1%})")
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        color, ls = STYLE[name]
        ax.plot(rec, prec, color=color, ls=ls, lw=2, label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall (share of defaults caught)")
    ax.set_ylabel("Precision (share of flagged loans that default)")
    ax.set_title("Precision-recall curves - validation split")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.45), fontsize=9)
    fig.savefig(path)
    return fig


def plot_single_model(name: str, y, p) -> plt.Figure:
    """ROC, PR and confusion matrix of one model (logged to its MLflow run)."""
    color, ls = STYLE[name]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)
    axes[0].plot(fpr, tpr, color=color, ls=ls, lw=2)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC (AUC {roc_auc_score(y, p):.3f})")
    prec, rec, _ = precision_recall_curve(y, p)
    axes[1].axhline(np.mean(y), color=MUTED, ls=":", lw=1)
    axes[1].plot(rec, prec, color=color, ls=ls, lw=2)
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall (PR-AUC {average_precision_score(y, p):.3f})")
    cm = confusion_matrix(y, (np.asarray(p) >= THRESHOLD).astype(int))
    axes[2].imshow(cm, cmap="Blues")
    axes[2].grid(False)
    for (i, j), v in np.ndenumerate(cm):
        axes[2].text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else INK, fontsize=11)
    axes[2].set_xticks([0, 1], ["Predicted repaid", "Predicted default"])
    axes[2].set_yticks([0, 1], ["Actual repaid", "Actual default"])
    axes[2].set_title(f"Confusion matrix (threshold {THRESHOLD})")
    fig.suptitle(f"{name} - validation split", x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT)


def run_one(name: str, pipe, d: dict, cv: StratifiedKFold) -> tuple[dict, np.ndarray]:
    X_train, y_train, X_val, y_val = d["X_train"], d["y_train"], d["X_val"], d["y_val"]

    # 1. Cross-validation on the training split (each fold refits the whole pipeline, incl. imputer and SMOTE)
    start = time.perf_counter()
    cv_scores = cross_val_score(clone(pipe), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    cv_seconds = time.perf_counter() - start

    # 2. Fit on the full training split, score the validation split
    start = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    val_proba = pipe.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])   # only to spot overfitting

    row = {
        "model": name,
        "imbalance_handling": IMBALANCE_HANDLING[name],
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std": cv_scores.std(),
        "train_auc": train_auc,
        **{f"val_{k}": v for k, v in evaluate(y_val, val_proba).items()},
        "fit_seconds": fit_seconds,
    }

    # 3. MLflow: one nested run per model
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params({
            "model_name": name,
            "preprocessor": "linear" if name == "Logistic Regression" else "tree",
            "imbalance_handling": IMBALANCE_HANDLING[name],
            "n_features_in": len(FEATURES),
            "n_train_rows": len(X_train),
            "cv_folds": CV_FOLDS,
            "threshold": THRESHOLD,
            **{f"model__{k}": v for k, v in model_params(pipe).items()},
        })
        mlflow.log_metrics({k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating))})
        mlflow.log_metrics({f"cv_auc_fold_{i + 1}": float(s) for i, s in enumerate(cv_scores)})
        mlflow.log_metric("cv_seconds", cv_seconds)
        fig = plot_single_model(name, y_val, val_proba)
        mlflow.log_figure(fig, "validation_plots.png")
        plt.close(fig)
        mlflow.sklearn.log_model(pipe, name="model", input_example=X_train.head(5),
                                 skops_trusted_types=SKOPS_TRUSTED_TYPES)

    print(f"  {name:20s} CV AUC {row['cv_auc_mean']:.4f} +/- {row['cv_auc_std']:.4f} | "
          f"val AUC {row['val_roc_auc']:.4f} | fit {fit_seconds:.1f}s, CV {cv_seconds:.1f}s")
    return row, val_proba


def recommendation_reasons(table: pd.DataFrame, best: str) -> list[str]:
    """Plain-language reasons for the recommendation, built from the real numbers."""
    t = table.set_index("model")
    b, lr = t.loc[best], t.loc["Logistic Regression"]
    xgb, smote, rf = t.loc["XGBoost"], t.loc["XGBoost + SMOTE"], t.loc["Random Forest"]
    reasons = []
    if best != "Logistic Regression":
        reasons.append(
            f"Tree models beat Logistic Regression by {b['val_roc_auc'] - lr['val_roc_auc']:.3f} validation ROC-AUC "
            f"({b['val_roc_auc']:.3f} vs {lr['val_roc_auc']:.3f}). They can learn the step patterns from the EDA "
            "(grade C -> D, loan_percent_income above 0.30, renters) and their combinations; a straight-line model cannot.")
    cv_gap = smote["cv_auc_mean"] - xgb["cv_auc_mean"]
    noise = max(xgb["cv_auc_std"], smote["cv_auc_std"])
    if abs(cv_gap) < noise:
        reasons.append(
            f"SMOTE does not clearly help: CV ROC-AUC {smote['cv_auc_mean']:.3f} vs {xgb['cv_auc_mean']:.3f} "
            f"(difference {cv_gap:+.3f}, smaller than one CV standard deviation of {noise:.3f}) and validation "
            f"{smote['val_roc_auc']:.3f} vs {xgb['val_roc_auc']:.3f}. scale_pos_weight is simpler, faster and does "
            "not create synthetic loans, so plain XGBoost is preferred.")
    reasons.append(
        f"Random Forest scores lower ({rf['val_roc_auc']:.3f}) and memorises the training data (train ROC-AUC "
        f"{rf['train_auc']:.3f}). XGBoost also overfits a little (train {xgb['train_auc']:.3f} vs validation "
        f"{xgb['val_roc_auc']:.3f}); tuning max_depth, min_child_weight and regularisation in Step 4 should help.")
    gaps = table["val_roc_auc"] - table["cv_auc_mean"]
    reasons.append(
        f"Validation ROC-AUC is {gaps.min():+.3f} to {gaps.max():+.3f} away from the CV mean for every model, so this "
        "validation split is slightly easier than average. Use the CV mean (not the validation number) as the tuning objective.")
    return reasons


def write_markdown(table: pd.DataFrame, best: str, rules: pd.DataFrame, path: Path) -> None:
    show = table[["model", "imbalance_handling", "cv_auc_mean", "cv_auc_std", "train_auc", "val_roc_auc",
                  "val_pr_auc", "val_ks", "val_precision", "val_recall", "val_f1"]]
    header = ["Model", "Imbalance handling", "CV ROC-AUC (mean)", "CV std", "Train ROC-AUC", "Val ROC-AUC",
              "Val PR-AUC", "Val KS", "Precision @0.5", "Recall @0.5", "F1 @0.5"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, r in show.iterrows():
        cells = [r["model"], r["imbalance_handling"]] + [f"{v:.3f}" for v in r.iloc[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    b = table.set_index("model").loc[best]
    text = [
        "# Model comparison - validation split",
        "",
        f"Made by `python src/train.py`. Training split: {table.attrs['n_train']:,} loans, validation split: "
        f"{table.attrs['n_val']:,} loans (default rate {table.attrs['val_rate']:.1%}). "
        f"Cross-validation: {CV_FOLDS}-fold stratified on the training split. Test split not used.",
        "",
        *lines,
        "",
        "Train ROC-AUC is shown only to spot overfitting (a big gap to the validation AUC = the model memorises).",
        f"Precision / recall / F1 use threshold {THRESHOLD}; the real decision cut-offs are chosen in Step 5.",
        "",
        "## Compared with the quick baselines (`reports/leakage_check.md`)",
        "",
        *[f"- {k}: {v:.3f} validation ROC-AUC" for k, v in BASELINES.items()],
        f"- Leakage alarm: {LEAKAGE_ALARM:.2f}. Highest validation ROC-AUC here: {table['val_roc_auc'].max():.3f} "
        f"-> {'below the alarm' if table['val_roc_auc'].max() <= LEAKAGE_ALARM else 'ABOVE THE ALARM - investigate'}.",
        "",
        "## Recommendation",
        "",
        f"Tune **{best}** in Step 4: highest validation ROC-AUC ({b['val_roc_auc']:.3f}), CV ROC-AUC "
        f"{b['cv_auc_mean']:.3f} +/- {b['cv_auc_std']:.3f}.",
        "",
        *[f"- {r}" for r in recommendation_reasons(table, best)],
        "",
        "## Why is the AUC so high? A perfect rule in the data",
        "",
        "The precision-recall chart (14) shows the tree models flag about two thirds of the defaulters with no",
        "false alarm at all. The reason, on the training split:",
        "",
        "| Rule | Loans | Defaults | Default rate |",
        "|---|---|---|---|",
        *[f"| {n} | {int(r['loans']):,} | {int(r['defaults']):,} | {r['default_rate']:.1%} |" for n, r in rules.iterrows()],
        "",
        "Every renter whose loan is more than 30% of income defaulted. This is not leakage (both values are known",
        "at application time), but a rule this clean is very unusual in real lending data and suggests the labels",
        "were at least partly generated by rules. So the AUC on this dataset is likely higher than the same model",
        "would reach on real loans - an honest limitation for the report.",
        "",
    ]
    path.write_text("\n".join(text))


def main() -> None:
    d = get_splits()
    setup_mlflow()
    FIG.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(d["y_train"])

    print(f"Training {len(models)} models: {CV_FOLDS}-fold CV on {len(d['X_train']):,} training rows, "
          f"then validation on {len(d['X_val']):,} rows")
    rows, curves = [], {}
    with mlflow.start_run(run_name="model_comparison"):
        for name, pipe in models.items():
            row, val_proba = run_one(name, pipe, d, cv)
            rows.append(row)
            curves[name] = (d["y_val"], val_proba)

        table = pd.DataFrame(rows).sort_values("val_roc_auc", ascending=False).reset_index(drop=True)
        table.attrs.update(n_train=len(d["X_train"]), n_val=len(d["X_val"]), val_rate=float(d["y_val"].mean()))
        best = table.loc[0, "model"]

        table.round(4).to_csv(REPORTS / "model_comparison.csv", index=False)
        write_markdown(table, best, perfect_rule_check(d["X_train"], d["y_train"]), REPORTS / "model_comparison.md")
        for fig in [plot_roc(curves, FIG / "13_roc_curves_validation.png"),
                    plot_pr(curves, FIG / "14_pr_curves_validation.png")]:
            plt.close(fig)

        mlflow.set_tag("best_model", best)
        mlflow.log_metric("best_val_roc_auc", float(table.loc[0, "val_roc_auc"]))
        for f in [REPORTS / "model_comparison.csv", REPORTS / "model_comparison.md",
                  FIG / "13_roc_curves_validation.png", FIG / "14_pr_curves_validation.png"]:
            mlflow.log_artifact(str(f))

    cols = ["model", "cv_auc_mean", "cv_auc_std", "train_auc", "val_roc_auc", "val_pr_auc", "val_ks",
            "val_precision", "val_recall", "val_f1"]
    print("\nModel comparison (validation split, sorted by ROC-AUC):")
    print(table[cols].round(3).to_string(index=False))
    print("\nBaselines from reports/leakage_check.md: " + ", ".join(f"{k} {v:.3f}" for k, v in BASELINES.items()))
    print(f"Recommended model to tune: {best}")
    print("Saved reports/model_comparison.csv, reports/model_comparison.md, figures 13 and 14; runs logged to MLflow")

    if (table["val_roc_auc"] > LEAKAGE_ALARM).any():
        too_high = table.loc[table["val_roc_auc"] > LEAKAGE_ALARM, "model"].tolist()
        raise SystemExit(f"STOP: validation ROC-AUC above {LEAKAGE_ALARM} for {too_high} - check for leakage "
                         "(python src/leakage_check.py) before tuning.")


if __name__ == "__main__":
    main()
