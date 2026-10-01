"""
Drift monitoring with Evidently (0.7 API: Report / Dataset / DataDefinition, presets from evidently.presets).

Reference = the training split. Current = a batch of new applications:
    - demo mode: a simulated batch from src/simulate_batches.py ("drift" or "no_drift")
    - live mode: recent applications stored in PostgreSQL (done by the API: app/services/drift_service.py)

Checks:
    1. data drift of the 11 model features (Evidently picks the test per column: Wasserstein distance for numbers,
       Jensen-Shannon distance for categories when the data is large)
    2. prediction drift: the saved model scores both sets; Evidently compares the predicted probabilities
       and we also compute the PSI (population stability index) of the predicted probability
Alert rule: drift_share > 0.3 (more than 30% of the features drifted) OR prediction PSI > 0.2 -> "DRIFT DETECTED".
Each run saves an HTML report + a JSON summary to reports/drift/ (timestamped, git-ignored).

Run the demo from the project folder (both simulated batches + reports/drift_summary.md):
    python src/drift.py
"""

import json
import os
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")   # Evidently: never send usage telemetry

import joblib
import numpy as np
import pandas as pd

from clean_data import to_model_input
from load_data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, get_splits

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "loan_default_model.joblib"
META_PATH = PROJECT_DIR / "models" / "model_meta.json"
DRIFT_DIR = Path(os.getenv("DRIFT_REPORT_DIR", PROJECT_DIR / "reports" / "drift"))   # the tests use a temp folder
SUMMARY_PATH = PROJECT_DIR / "reports" / "drift_summary.md"

DRIFT_SHARE_ALARM = 0.3   # more than 30% of the 11 features drifted
PSI_ALARM = 0.2           # PSI above 0.2 = a large shift (rule of thumb used by banks)
PREDICTION = "predicted_probability"


# ---------------------------------------------------------------------------
# Small, testable building blocks
# ---------------------------------------------------------------------------

def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index: how much the distribution of `current` moved away from `reference`.

    Bins = deciles of the reference (bins that are identical because of tied values are merged). For each bin:
    (current share - reference share) x ln(current share / reference share), summed. 0 = same distribution;
    below 0.1 small, 0.1-0.2 moderate, above 0.2 large shift."""
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.clip(np.histogram(reference, edges)[0] / len(reference), eps, None)
    cur_share = np.clip(np.histogram(current, edges)[0] / len(current), eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def drift_status(drift_share: float, prediction_psi: float) -> str:
    """The alert rule."""
    return "DRIFT DETECTED" if drift_share > DRIFT_SHARE_ALARM or prediction_psi > PSI_ALARM else "NO DRIFT"


def column_drifted(score: float, method: str, threshold: float) -> bool:
    """Evidently's rule: for p-value tests drift = p < threshold, for distances drift = distance >= threshold."""
    return score < threshold if "p_value" in method else score >= threshold


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _reference() -> tuple[pd.DataFrame, np.ndarray]:
    """Training split features + the saved model's predicted probabilities for them (computed once)."""
    X = get_splits()["X_train"].reset_index(drop=True)
    return X, _model().predict_proba(X)[:, 1]


def _cutoffs() -> dict:
    return json.loads(META_PATH.read_text())["decision_cutoffs"]


def run_drift_check(current: pd.DataFrame, batch_name: str, save: bool = True) -> dict:
    """Compare a batch of applications (the 11 raw fields: grade letter, Y/N) with the training data.
    Returns a summary dict; with save=True also writes the HTML report and JSON summary to reports/drift/."""
    from evidently import DataDefinition, Dataset, Report     # imported here: only needed when a check runs
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset

    ref_X, ref_p = _reference()
    cur_X = to_model_input(current.drop(columns=["loan_id"], errors="ignore")).reset_index(drop=True)
    cur_p = _model().predict_proba(cur_X)[:, 1]

    definition = DataDefinition(numerical_columns=NUMERIC_FEATURES + [PREDICTION], categorical_columns=CATEGORICAL_FEATURES)
    reference = Dataset.from_pandas(ref_X.assign(**{PREDICTION: ref_p}), data_definition=definition)
    current_ds = Dataset.from_pandas(cur_X.assign(**{PREDICTION: cur_p}), data_definition=definition)
    report = Report([DataDriftPreset(columns=FEATURES, drift_share=DRIFT_SHARE_ALARM), ValueDrift(column=PREDICTION)])
    snapshot = report.run(current_data=current_ds, reference_data=reference)

    features, prediction_drift, evidently_count = {}, None, None
    for m in snapshot.dict()["metrics"]:
        cfg = m["config"]
        if cfg["type"].endswith("DriftedColumnsCount"):
            evidently_count = int(m["value"]["count"])
        elif cfg["type"].endswith("ValueDrift"):
            item = {"score": float(m["value"]), "method": cfg["method"], "threshold": float(cfg["threshold"])}
            item["drifted"] = column_drifted(item["score"], item["method"], item["threshold"])
            if cfg["column"] == PREDICTION:
                prediction_drift = item
            else:
                features[cfg["column"]] = item
    drifted = [f for f, v in features.items() if v["drifted"]]
    if evidently_count is not None and evidently_count != len(drifted):
        raise RuntimeError(f"drift count mismatch: Evidently {evidently_count}, ours {len(drifted)}")

    cut = _cutoffs()
    drift_share = len(drifted) / len(FEATURES)
    prediction_psi = psi(ref_p, cur_p)
    created = datetime.now(timezone.utc)
    summary = {
        "created_at": created.isoformat(timespec="seconds"),
        "batch_name": batch_name,
        "n_reference": len(ref_X),
        "n_current": len(cur_X),
        "drift_share": drift_share,
        "n_drifted": len(drifted),
        "drifted_features": sorted(drifted, key=lambda f: -features[f]["score"]),
        "dataset_drift": drift_share > DRIFT_SHARE_ALARM,
        "prediction_psi": prediction_psi,
        "prediction_drift": prediction_drift,
        "mean_probability": {"reference": float(ref_p.mean()), "current": float(cur_p.mean())},
        "reject_share": {"reference": float(np.mean(ref_p >= cut["reject"])), "current": float(np.mean(cur_p >= cut["reject"]))},
        "status": drift_status(drift_share, prediction_psi),
        "features": features,
        "rules": {"drift_share_alarm": DRIFT_SHARE_ALARM, "psi_alarm": PSI_ALARM},
        "report_path": None,
    }
    if save:
        DRIFT_DIR.mkdir(parents=True, exist_ok=True)
        stem = DRIFT_DIR / f"drift_{created:%Y%m%d_%H%M%S}_{batch_name}"
        snapshot.save_html(str(stem.with_suffix(".html")))
        html = stem.with_suffix(".html")
        summary["report_path"] = str(html.relative_to(PROJECT_DIR) if html.is_relative_to(PROJECT_DIR) else html)
        stem.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Demo: both simulated batches
# ---------------------------------------------------------------------------

def write_summary(results: dict[str, dict]) -> None:
    """reports/drift_summary.md: the two batches side by side (for the report)."""
    nd, dr = results["no_drift"], results["drift"]
    rows = [f"| {f} | {nd['features'][f]['method']} | {nd['features'][f]['score']:.3f}"
            f"{' **drift**' if nd['features'][f]['drifted'] else ''} | {dr['features'][f]['score']:.3f}"
            f"{' **drift**' if dr['features'][f]['drifted'] else ''} | {dr['features'][f]['threshold']:g} |"
            for f in sorted(FEATURES, key=lambda f: -dr["features"][f]["score"])]
    lines = [
        "# Drift monitoring - simulated batches",
        "",
        "Made by `python src/drift.py`. The dataset has no dates, so drift is **simulated**: two batches of 2,000 loans",
        "from the test split (`src/simulate_batches.py`). Reference = the training split "
        f"({nd['n_reference']:,} loans). Alert rule: more than {DRIFT_SHARE_ALARM:.0%} of the features drifted, or",
        f"prediction PSI above {PSI_ALARM}.",
        "",
        "| | no_drift batch | drift batch |",
        "|---|---|---|",
        f"| Features drifted | {nd['n_drifted']} of {len(FEATURES)} ({nd['drift_share']:.0%}) | "
        f"{dr['n_drifted']} of {len(FEATURES)} ({dr['drift_share']:.0%}) |",
        f"| Drifted features | {', '.join(nd['drifted_features']) or '-'} | {', '.join(dr['drifted_features']) or '-'} |",
        f"| Prediction PSI | {nd['prediction_psi']:.3f} | {dr['prediction_psi']:.3f} |",
        f"| Mean predicted probability of default (reference {nd['mean_probability']['reference']:.1%}) | "
        f"{nd['mean_probability']['current']:.1%} | {dr['mean_probability']['current']:.1%} |",
        f"| Share in the REJECT band (reference {nd['reject_share']['reference']:.1%}) | "
        f"{nd['reject_share']['current']:.1%} | {dr['reject_share']['current']:.1%} |",
        f"| **Status** | **{nd['status']}** | **{dr['status']}** |",
        "",
        "## Drift score per feature",
        "",
        "Higher score = bigger difference from the training data; a feature drifts when its score reaches the threshold.",
        "",
        "| Feature | Test used | no_drift score | drift score | Threshold |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "The drift batch was built with: interest rates +3 points, incomes -15% (so loan / income goes up), and debt",
        "consolidation + medical loans raised from about 35% to 50% of the batch. Interactive HTML reports: `reports/drift/`.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines))


def main() -> None:
    from simulate_batches import load_batch

    results = {}
    for name in ["no_drift", "drift"]:
        results[name] = r = run_drift_check(load_batch(name), batch_name=name)
        print(f"{name:9s} -> {r['status']:15s} drifted {r['n_drifted']}/{len(FEATURES)} ({r['drift_share']:.0%}): "
              f"{', '.join(r['drifted_features']) or '-'} | prediction PSI {r['prediction_psi']:.3f} | "
              f"mean probability {r['mean_probability']['reference']:.1%} -> {r['mean_probability']['current']:.1%}")
        print(f"          report: {r['report_path']}")
    write_summary(results)
    print(f"Saved {SUMMARY_PATH.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
