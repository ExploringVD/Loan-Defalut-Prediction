"""Daily drift check (live mode) with APScheduler, started by the FastAPI app.

Runs every day at 02:00 (server time). Turned off when DRIFT_SCHEDULER=0 (the tests do this)."""
import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.db import SessionLocal

logger = logging.getLogger("uvicorn.error")
JOB_ID = "daily_drift_check"


def daily_drift_job() -> None:
    from app.services.drift_service import run_and_store
    try:
        with SessionLocal() as db:
            run_and_store(db, "live")
    except Exception:   # noqa: BLE001 - a failed check must never stop the API
        logger.exception("Daily drift check failed")


def start_scheduler() -> BackgroundScheduler | None:
    if os.getenv("DRIFT_SCHEDULER", "1") != "1":
        return None
    scheduler = BackgroundScheduler()
    scheduler.add_job(daily_drift_job, CronTrigger(hour=2, minute=0), id=JOB_ID, replace_existing=True)
    scheduler.start()
    logger.info("Drift check scheduled daily at 02:00 (next run %s)", scheduler.get_job(JOB_ID).next_run_time)
    return scheduler
