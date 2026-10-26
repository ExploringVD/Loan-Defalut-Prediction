"""
Batch scoring: score a CSV of loan applicants with the production model.

    python src/predict_batch.py --input data/simulated/drift_batch.csv --output reports/batch_predictions.csv

Input: the 11 raw applicant fields (same column names as the Kaggle file). Optional columns: loan_id,
applicant_name and loan_status (the real outcome - if present, ROC-AUC and the default rate per band are printed).
Every row is checked with the SAME rules as the API (applicant_schema.ApplicantIn). Invalid rows are not scored:
they go to <output>_rejected.csv with the reason.
Output: loan_id / applicant_name (if given), probability, decision band (cut-offs from model_meta.json) and the
top reason in plain English (SHAP).

score_batch() is the one reusable function - the API (POST /applications/batch) calls it too.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.metrics import roc_auc_score

from applicant_schema import ApplicantIn
from clean_data import APPLICANT_FIELDS, OPTIONAL_FIELDS, to_model_input
from decision import BANDS, assign_bands
from explain import RAW_FIELD, get_explainer, reason_text

PROJECT_DIR = Path(__file__).resolve().parents[1]
ID_COLUMNS = ["loan_id", "applicant_name"]          # passed through to the output when present
TARGET = "loan_status"
REQUIRED_COLUMNS = [c for c in APPLICANT_FIELDS if c not in OPTIONAL_FIELDS]
NO_NAME = "batch row"                              # ApplicantIn needs a name; used only when the CSV has none


class BatchError(ValueError):
    """A problem with the whole file (unreadable, empty, missing columns) - not with single rows."""


def read_applicants_csv(source) -> pd.DataFrame:
    """CSV (path or file object) -> DataFrame. Stops with a BatchError if the file itself is unusable."""
    try:
        df = pd.read_csv(source, dtype={"applicant_name": "string"})
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise BatchError(f"Could not read the CSV file: {e}") from None
    if df.empty:
        raise BatchError("The CSV file has no rows.")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise BatchError(f"The CSV file is missing these columns: {', '.join(missing)}")
    return df


def error_text(error: ValidationError) -> str:
    """Pydantic errors -> one readable line, e.g. 'person_age: Input should be greater than or equal to 18'."""
    parts = []
    for e in error.errors():
        message = str(e["msg"]).removeprefix("Value error, ")
        field = ".".join(str(x) for x in e["loc"])
        parts.append(f"{field}: {message}" if field else message)
    return "; ".join(parts)


def validate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check every row with the API's rules. Returns (valid rows with clean values, rejected rows with a reason).
    'row' is the row number in the file (1 = first row under the header)."""
    valid, rejected = [], []
    has_name = "applicant_name" in df.columns
    df = df.astype(object).where(df.notna(), None)      # empty cells / NaN -> None = "not given"
    for i, record in enumerate(df.to_dict("records"), start=1):
        data = {f: record.get(f) for f in APPLICANT_FIELDS}
        data["applicant_name"] = record.get("applicant_name") if has_name else NO_NAME
        extra = {c: record.get(c) for c in [*ID_COLUMNS, TARGET] if c in df.columns}
        try:
            applicant = ApplicantIn.model_validate(data)
        except ValidationError as e:
            rejected.append({"row": i, **{c: extra[c] for c in ID_COLUMNS if c in extra},
                             "reason": error_text(e), **{f: record.get(f) for f in APPLICANT_FIELDS}})
            continue
        valid.append({"row": i, **extra, **applicant.applicant_fields()})
    reject_columns = ["row", *[c for c in ID_COLUMNS if c in df.columns], "reason", *APPLICANT_FIELDS]
    return pd.DataFrame(valid), pd.DataFrame(rejected, columns=reject_columns)


def top_reasons(valid: pd.DataFrame, shap_values: pd.DataFrame) -> list[str]:
    """The strongest SHAP reason of every row as a sentence, e.g. 'Grade D loan - increases risk'."""
    texts = []
    for (_, row), (_, values) in zip(valid.iterrows(), shap_values.iterrows()):
        feature = values.abs().idxmax()
        texts.append(reason_text([{"feature": feature, "value": row[RAW_FIELD[feature]],
                                   "direction": "increases risk" if values[feature] > 0 else "decreases risk"}])[0])
    return texts


def evaluate(scored: pd.DataFrame) -> dict | None:
    """ROC-AUC and the real default rate per band - only for rows that have loan_status 0 / 1."""
    if TARGET not in scored.columns:
        return None
    labelled = scored[pd.to_numeric(scored[TARGET], errors="coerce").isin([0, 1])]
    if labelled.empty:
        return None
    y = labelled[TARGET].astype(int)
    by_band = labelled.assign(y=y).groupby("decision")["y"].agg(["size", "mean"]).reindex(BANDS)
    return {
        "rows_with_loan_status": int(len(labelled)),
        "roc_auc": float(roc_auc_score(y, labelled["probability"])) if y.nunique() == 2 else None,
        "default_rate_by_band": {b: (None if pd.isna(r["mean"]) else float(r["mean"])) for b, r in by_band.iterrows()},
        "rows_by_band": {b: int(0 if pd.isna(r["size"]) else r["size"]) for b, r in by_band.iterrows()},
    }


def score_batch(df: pd.DataFrame) -> dict:
    """Validate and score many applicants at once (the same model, conversion and cut-offs as the API).

    Returns {"scored": DataFrame, "rejected": DataFrame, "summary": dict}."""
    start = time.perf_counter()
    explainer = get_explainer()
    cutoffs = explainer.cutoffs
    valid, rejected = validate_rows(df)
    out_columns = [c for c in ID_COLUMNS if c in df.columns] + ["probability", "decision", "top_reason"]
    out_columns += [TARGET] if TARGET in df.columns else []

    if valid.empty:
        scored = pd.DataFrame(columns=["row", *out_columns])
    else:
        X = to_model_input(valid[APPLICANT_FIELDS])               # the ONE raw -> model conversion
        probability = explainer.predict_proba(X)                  # calibrated probability of default
        shap_values = explainer.shap_by_feature(X)                # one SHAP pass for all rows
        scored = valid.assign(probability=np.round(probability, 4),
                              decision=assign_bands(probability, cutoffs["review"], cutoffs["reject"]),
                              top_reason=top_reasons(valid, shap_values))[["row", *out_columns]]

    decisions = scored["decision"].value_counts()
    summary = {
        "rows_total": len(df),
        "rows_scored": len(scored),
        "rows_rejected": len(rejected),
        "decisions": {b: int(decisions.get(b, 0)) for b in BANDS},
        "cutoffs": {"review": cutoffs["review"], "reject": cutoffs["reject"]},
        "model_version": explainer.meta["version"],
        "seconds": round(time.perf_counter() - start, 2),
        "evaluation": evaluate(scored),
    }
    return {"scored": scored, "rejected": rejected, "summary": summary}


def print_summary(summary: dict) -> None:
    s = summary
    print(f"Rows in file: {s['rows_total']:,} | scored: {s['rows_scored']:,} | rejected: {s['rows_rejected']:,} "
          f"| time: {s['seconds']:.1f} s (model version {s['model_version']})")
    cut = s["cutoffs"]
    print(f"Decision mix (APPROVE < {cut['review']}, REVIEW {cut['review']}-{cut['reject']}, REJECT >= {cut['reject']}):")
    for band, n in s["decisions"].items():
        share = n / s["rows_scored"] if s["rows_scored"] else 0
        print(f"  {band:8s} {n:6,d} ({share:.1%})")
    ev = s["evaluation"]
    if ev:
        auc = "n/a (needs both outcomes)" if ev["roc_auc"] is None else f"{ev['roc_auc']:.4f}"
        print(f"loan_status given for {ev['rows_with_loan_status']:,} rows -> ROC-AUC {auc}")
        for band, rate in ev["default_rate_by_band"].items():
            print(f"  {band:8s} real default rate {'-' if rate is None else f'{rate:.1%}'} "
                  f"({ev['rows_by_band'][band]:,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a CSV of loan applicants with the production model.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with the 11 applicant fields")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "reports" / "batch_predictions.csv",
                        help="scored rows (invalid rows go to <output>_rejected.csv)")
    args = parser.parse_args()

    try:
        result = score_batch(read_applicants_csv(args.input))
    except (BatchError, FileNotFoundError) as e:
        raise SystemExit(f"Error: {e}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = args.output.with_name(f"{args.output.stem}_rejected{args.output.suffix}")
    result["scored"].to_csv(args.output, index=False)
    result["rejected"].to_csv(rejected_path, index=False)
    print_summary(result["summary"])
    print(f"Saved {args.output} ({len(result['scored']):,} rows)")
    print(f"Saved {rejected_path} ({len(result['rejected']):,} rows)")


if __name__ == "__main__":
    main()
