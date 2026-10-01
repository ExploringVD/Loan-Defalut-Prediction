"""Admin-only monitoring: usage statistics and data drift checks."""
from datetime import datetime, timedelta, timezone
from typing import Literal

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import require_admin
from app.db import get_db
from app.models import DriftReport, Prediction, User
from app.schemas import DayStats, DriftReportOut, DriftRunOut, StatsOut
from app.services.drift_service import run_and_store

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/stats", response_model=StatsOut)
def stats(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    latencies = db.scalars(select(Prediction.latency_ms)).all()
    decisions = dict(db.execute(select(Prediction.decision, func.count()).group_by(Prediction.decision)).all())

    # Last 7 days (today included), counted in Python so it works the same on PostgreSQL and SQLite
    today = datetime.now(timezone.utc).date()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    start = datetime.combine(days[0], datetime.min.time(), tzinfo=timezone.utc)
    recent = db.execute(select(Prediction.created_at, Prediction.decision).where(Prediction.created_at >= start)).all()
    per_day = {d: {"requests": 0, "approve": 0, "review": 0, "reject": 0} for d in days}
    for created_at, decision in recent:
        day = created_at.date()
        if day in per_day:
            per_day[day]["requests"] += 1
            per_day[day][decision.lower()] += 1

    return StatsOut(
        total_requests=len(latencies),
        avg_latency_ms=round(float(np.mean(latencies)), 2) if latencies else None,
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2) if latencies else None,
        decisions={d: decisions.get(d, 0) for d in ["APPROVE", "REVIEW", "REJECT"]},
        last_7_days=[DayStats(date=d.isoformat(), **counts) for d, counts in per_day.items()],
    )


@router.post("/drift/run", response_model=DriftRunOut)
def run_drift(batch: Literal["live", "drift", "no_drift"] = Query("live", description=(
                  "live = applications of the last 30 days (needs at least 50); "
                  "drift / no_drift = the simulated demo batches")),
              db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    """Run a drift check now (takes a few seconds). The result is stored and shown on the Dashboard."""
    return run_and_store(db, batch, username=admin.username)


@router.get("/drift/latest", response_model=DriftReportOut)
def latest_drift(db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    """The most recent drift check; 404 if none has run yet."""
    report = db.scalar(select(DriftReport).order_by(DriftReport.id.desc()).limit(1))
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No drift check has run yet")
    return report
