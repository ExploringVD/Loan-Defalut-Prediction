# FastAPI backend: model + SHAP explainer + drift checks.
#   docker compose up --build        (see docker-compose.yml; needs .env)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DO_NOT_TRACK=1

# XGBoost needs the OpenMP runtime, which the slim image does not have
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Libraries first, so changing the code does not reinstall them
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Only what the API needs. The parquet file is the drift reference (training split) and the source of the
# simulated drift batches.
COPY alembic.ini ./
COPY alembic/ alembic/
COPY src/ src/
COPY app/ app/
COPY models/loan_default_model.joblib models/model_meta.json models/
COPY data/processed/credit_clean.parquet data/processed/

# Code stays owned by root but readable for everyone (some files are owner-only on the host and COPY keeps that).
# Folders the API writes to: drift reports (reports/ is mounted by docker compose) and the simulated batches.
RUN chmod -R a+rX /app \
    && mkdir -p reports/drift data/simulated && chown -R appuser:appuser reports data/simulated

USER appuser
ENV DRIFT_SCHEDULER=1
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"

# Create / update the tables, create the admin + officer users from .env (safe to repeat), then start the API
CMD ["sh", "-c", "alembic upgrade head && python -m app.seed && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
