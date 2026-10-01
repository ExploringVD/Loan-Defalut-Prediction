"""Drift checks for the API: pick the batch (live applications or a simulated batch), run src/drift.py,
store the result in drift_reports and the audit log."""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AuditLog, DriftReport, LoanApplication
from clean_data import APPLICANT_FIELDS

logger = logging.getLogger("uvicorn.error")
MIN_LIVE_APPLICATIONS = 50   # fewer stored applications -> the comparison would be noise, so skip
LIVE_DAYS = 30               # live mode uses the applications of the last 30 days


def live_batch(db: Session) -> pd.DataFrame:
    """Applications stored in the last LIVE_DAYS days, as the 11 raw fields."""
    since = datetime.now(timezone.utc) - timedelta(days=LIVE_DAYS)
    rows = db.scalars(select(LoanApplication).where(LoanApplication.created_at >= since)).all()
    return pd.DataFrame([{f: getattr(r, f) for f in APPLICANT_FIELDS} for r in rows], columns=APPLICANT_FIELDS)


def run_and_store(db: Session, batch: str, username: str = "scheduler") -> dict:
    """Run one drift check. batch = 'live', 'drift' or 'no_drift'. Returns the summary (+ the stored row id),
    or {'status': 'SKIPPED', ...} when live mode has too few applications."""
    from drift import run_drift_check            # imported here so the API starts fast (Evidently is big)
    from simulate_batches import load_batch

    if batch == "live":
        current = live_batch(db)
        if len(current) < MIN_LIVE_APPLICATIONS:
            message = (f"Skipped: only {len(current)} application(s) stored in the last {LIVE_DAYS} days; "
                       f"a drift check needs at least {MIN_LIVE_APPLICATIONS}.")
            logger.info("Drift check (live): %s", message)
            return {"status": "SKIPPED", "batch_name": "live", "message": message, "n_current": len(current)}
    else:
        current = load_batch(batch)

    summary = run_drift_check(current, batch_name=batch)
    row = DriftReport(batch_name=batch, drift_share=summary["drift_share"], dataset_drift=summary["dataset_drift"],
                      prediction_psi=summary["prediction_psi"], status=summary["status"],
                      report_path=summary["report_path"])
    db.add(row)
    db.add(AuditLog(username=username, action="drift_check",
                    details={"batch": batch, "status": summary["status"], "drift_share": summary["drift_share"],
                             "prediction_psi": summary["prediction_psi"]}))
    db.commit()
    level = logging.WARNING if summary["status"] == "DRIFT DETECTED" else logging.INFO
    logger.log(level, "Drift check (%s): %s - %d/11 features drifted, prediction PSI %.3f", batch,
               summary["status"], summary["n_drifted"], summary["prediction_psi"])
    return {**summary, "id": row.id, "created_at": row.created_at.isoformat(), "message": None}
