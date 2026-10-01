"""GET /monitoring/stats (admin only): how much the API is used, how fast it is, and the decision mix."""
from datetime import datetime, timedelta, timezone

import numpy as np
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import require_admin
from app.db import get_db
from app.models import Prediction, User
from app.schemas import DayStats, StatsOut

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
