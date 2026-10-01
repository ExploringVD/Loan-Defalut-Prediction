"""FastAPI app. Start it from the project folder:
    uvicorn app.main:app --reload
Swagger docs: http://127.0.0.1:8000/docs
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import applications, auth, model, monitoring
from app.scheduler import start_scheduler
from app.services.scoring import load_model, model_meta

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model + SHAP explainer ONCE, before the first request
    seconds = load_model()
    app.state.model_load_seconds = round(seconds, 2)
    logger.info("Model version %s and SHAP explainer loaded in %.1f s", model_meta()["version"], seconds)
    scheduler = start_scheduler()          # daily drift check at 02:00 (off when DRIFT_SCHEDULER=0)
    yield
    if scheduler:
        scheduler.shutdown(wait=False)


app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "Scores loan applications with a calibrated XGBoost model and explains every decision with SHAP.\n\n"
        "1. `POST /auth/login` (or the **Authorize** button) with your username and password.\n"
        "2. `POST /applications` with the 11 applicant fields -> probability of default, APPROVE / REVIEW / "
        "REJECT and the top 5 reasons."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    """Every response tells how long the whole request took (header X-Process-Time-Ms)."""
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


@app.get("/health", tags=["health"])
def health(db: Session = Depends(get_db)) -> dict:
    """Public check: is the API up, is the model loaded, can we reach the database?"""
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:   # noqa: BLE001 - report, do not crash the health check
        database = "unreachable"
    return {"status": "ok" if database == "ok" else "degraded", "model_version": model_meta()["version"],
            "model_load_seconds": getattr(app.state, "model_load_seconds", None), "database": database}


for r in (auth.router, applications.router, model.router, monitoring.router):
    app.include_router(r)
