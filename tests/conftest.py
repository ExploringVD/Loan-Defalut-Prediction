"""Shared test set-up.

The API settings are set through environment variables here, BEFORE any test imports the app, so tests never use
the real database or passwords from .env. The API fixtures build a temporary SQLite database with the real Alembic
migration and start the app once per test session (the model + explainer load takes a few seconds).
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
TMP_DIR = tempfile.mkdtemp(prefix="loan_api_test_")
os.environ.update({
    "DATABASE_URL": f"sqlite:///{TMP_DIR}/test.db",
    "JWT_SECRET": "test-secret-not-used-anywhere-else",
    "JWT_EXPIRE_MINUTES": "5",
    "ADMIN_USERNAME": "test_admin", "ADMIN_PASSWORD": "admin-pass",
    "OFFICER_USERNAME": "test_officer", "OFFICER_PASSWORD": "officer-pass",
})
for path in (PROJECT_DIR, PROJECT_DIR / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
# frontend/ goes LAST: it has its own app.py, which must not hide the backend package app/
if str(PROJECT_DIR / "frontend") not in sys.path:
    sys.path.append(str(PROJECT_DIR / "frontend"))

MODEL_READY = (PROJECT_DIR / "models" / "loan_default_model.joblib").exists()
needs_model = pytest.mark.skipif(not MODEL_READY, reason="run python src/tune.py and src/business.py first")


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient on a fresh SQLite database (migrated with Alembic, admin + officer seeded)."""
    if not MODEL_READY:
        pytest.skip("run python src/tune.py and src/business.py first")
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient

    from app.db import SessionLocal
    from app.main import app
    from app.seed import seed

    cfg = Config(str(PROJECT_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")
    with SessionLocal() as db:
        seed(db)
    with TestClient(app) as c:
        yield c


def login_headers(client, username, password) -> dict:
    r = client.post("/auth/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
def admin(client):
    return login_headers(client, "test_admin", "admin-pass")


@pytest.fixture(scope="session")
def officer(client):
    return login_headers(client, "test_officer", "officer-pass")
