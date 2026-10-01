"""API tests with a temporary SQLite database (no Docker / PostgreSQL needed).

The settings are set through environment variables BEFORE the app is imported, so the tests never use the
real database or passwords from .env. The tables are created with the real Alembic migration.
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
sys.path.insert(0, str(PROJECT_DIR))

MODEL_READY = (PROJECT_DIR / "models" / "loan_default_model.joblib").exists()
pytestmark = pytest.mark.skipif(not MODEL_READY, reason="run python src/tune.py and src/business.py first")

if MODEL_READY:
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient

    from app.db import SessionLocal
    from app.main import app
    from app.models import AuditLog
    from app.schemas import EXAMPLES
    from app.seed import seed

LOW = EXAMPLES["low_risk"]["value"] if MODEL_READY else {}
HIGH = EXAMPLES["high_risk"]["value"] if MODEL_READY else {}


@pytest.fixture(scope="module")
def client():
    cfg = Config(str(PROJECT_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
    command.upgrade(cfg, "head")                      # the real migration, on SQLite
    with SessionLocal() as db:
        seed(db)
    with TestClient(app) as c:                         # runs the startup: model + explainer loaded once
        yield c


def token(client, username, password):
    r = client.post("/auth/login", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def admin(client):
    return token(client, "test_admin", "admin-pass")


@pytest.fixture(scope="module")
def officer(client):
    return token(client, "test_officer", "officer-pass")


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["database"] == "ok"


def test_login_and_roles(client, admin, officer):
    assert client.get("/auth/me", headers=admin).json() == {"username": "test_admin", "role": "admin"}
    assert client.get("/auth/me", headers=officer).json()["role"] == "officer"


def test_wrong_password_is_rejected_and_logged(client):
    r = client.post("/auth/login", data={"username": "test_admin", "password": "wrong"})
    assert r.status_code == 401
    with SessionLocal() as db:
        assert db.query(AuditLog).filter_by(username="test_admin", action="login_failed").count() >= 1


@pytest.mark.parametrize("method,url", [("post", "/applications"), ("get", "/applications"),
                                        ("get", "/model/info"), ("get", "/monitoring/stats")])
def test_auth_required(client, method, url):
    r = client.request(method, url, json=LOW if method == "post" else None)
    assert r.status_code == 401


def test_bad_token_is_rejected(client):
    r = client.get("/applications", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_score_low_and_high_risk(client, officer):
    r = client.post("/applications", json=LOW, headers=officer)
    assert r.status_code == 201, r.text
    body = r.json()
    p = body["prediction"]
    assert p["decision"] == "APPROVE" and 0 <= p["probability"] < 0.10
    assert len(p["top_reasons"]) == 5 and all(x["text"].endswith("risk") for x in p["top_reasons"])
    assert p["latency_ms"] < 1000 and "x-process-time-ms" in r.headers
    assert body["created_by"] == "test_officer" and body["loan_grade"] == "A"

    r = client.post("/applications", json=HIGH, headers=officer)
    assert r.status_code == 201
    assert r.json()["prediction"]["decision"] == "REJECT"


def test_optional_fields_can_be_missing(client, officer):
    body = {k: v for k, v in LOW.items() if k not in ("loan_int_rate", "person_emp_length")}
    r = client.post("/applications", json=body, headers=officer)
    assert r.status_code == 201
    assert r.json()["loan_int_rate"] is None


def test_lower_case_categories_are_accepted(client, officer):
    r = client.post("/applications", json={**LOW, "loan_grade": "a", "person_home_ownership": "rent"}, headers=officer)
    assert r.status_code == 201 and r.json()["loan_grade"] == "A"


@pytest.mark.parametrize("change", [
    {"person_age": 144},                                  # impossible age
    {"loan_grade": "Z"},                                  # not a grade
    {"person_age": 22, "person_emp_length": 10},          # employed longer than the age allows
    {"cb_person_default_on_file": "maybe"},
    {"loan_intent": "HOLIDAY"},
    {"person_income": 0},
    {"loan_percent_income": 0.6},                         # does not match loan_amnt / person_income
    {"cb_person_cred_hist_length": 50},                   # longer than the age
    {"unknown_field": 1},                                 # extra fields are not allowed
])
def test_invalid_input_returns_422(client, officer, change):
    r = client.post("/applications", json={**LOW, **change}, headers=officer)
    assert r.status_code == 422, r.text


def test_missing_required_field_returns_422(client, officer):
    body = {k: v for k, v in LOW.items() if k != "person_income"}
    assert client.post("/applications", json=body, headers=officer).status_code == 422


def test_list_filter_and_get(client, officer):
    client.post("/applications", json=HIGH, headers=officer)
    page = client.get("/applications", params={"decision": "REJECT", "page_size": 1}, headers=officer).json()
    assert page["total"] >= 1 and len(page["items"]) == 1
    assert all(item["prediction"]["decision"] == "REJECT" for item in page["items"])
    app_id = page["items"][0]["id"]
    assert client.get(f"/applications/{app_id}", headers=officer).json()["id"] == app_id
    assert client.get("/applications/999999", headers=officer).status_code == 404
    assert client.get("/applications", params={"decision": "MAYBE"}, headers=officer).status_code == 422


def test_scoring_writes_audit_rows(client, officer):
    client.post("/applications", json=LOW, headers=officer)
    with SessionLocal() as db:
        assert db.query(AuditLog).filter_by(username="test_officer", action="score_application").count() >= 1


def test_officer_cannot_see_stats_admin_can(client, admin, officer):
    assert client.get("/monitoring/stats", headers=officer).status_code == 403
    r = client.get("/monitoring/stats", headers=admin)
    assert r.status_code == 200
    stats = r.json()
    assert stats["total_requests"] >= 2
    assert stats["decisions"]["APPROVE"] >= 1 and stats["decisions"]["REJECT"] >= 1
    assert len(stats["last_7_days"]) == 7 and stats["last_7_days"][-1]["requests"] >= 2


def test_model_info(client, officer):
    info = client.get("/model/info", headers=officer).json()
    assert info["decision_cutoffs"] == {"review": 0.1, "reject": 0.35}
    assert 0.85 <= info["test_metrics"]["roc_auc"] <= 0.97
    assert len(info["features"]) == 11


def test_swagger_has_examples(client):
    spec = client.get("/openapi.json").json()
    examples = spec["paths"]["/applications"]["post"]["requestBody"]["content"]["application/json"]["examples"]
    assert {"low_risk", "high_risk"} <= set(examples)
