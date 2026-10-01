"""Frontend tests: helpers, error handling, and the real Streamlit app run headless with AppTest.

The app is wired to the real FastAPI app (FastAPI's TestClient on the temporary SQLite database from conftest.py)
by replacing api.make_http_client - so these tests go frontend -> API -> model -> database, without a server.
"""
import httpx
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import api
import ui
from conftest import PROJECT_DIR, needs_model
from views.new_application import DEFAULTS, LIMITS

APP_FILE = str(PROJECT_DIR / "frontend" / "app.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p,text", [(0.0, "< 1%"), (0.0049, "< 1%"), (0.2, "20.0%"), (0.996, "> 99%"), (1.0, "> 99%"),
                                    (None, "-")])
def test_format_probability(p, text):
    assert ui.format_probability(p) == text


def test_every_decision_has_colour_icon_and_words():
    for d in ["APPROVE", "REVIEW", "REJECT"]:
        assert {"icon", "kind", "color", "text"} <= set(ui.DECISIONS[d])
        assert d in ui.decision_label(d)


def test_parse_validation_errors_maps_to_fields():
    detail = [
        {"loc": ["body", "person_age"], "msg": "Input should be less than or equal to 100"},
        {"loc": ["body"], "msg": "Value error, person_emp_length cannot be more than person_age - 14"},
        {"loc": ["body"], "msg": "Value error, something general"},
    ]
    fields, general = api.parse_validation_errors(detail)
    assert fields == {"person_age": "Input should be less than or equal to 100",
                      "person_emp_length": "Years employed cannot be more than age - 14"}
    assert general == ["Something general"]


def test_reasons_figure_colours_and_order():
    reasons = [{"feature": "loan_grade_num", "label": "Loan grade", "value": "D", "contribution": 1.5,
                "direction": "increases risk", "text": "Grade D loan - increases risk"},
               {"feature": "person_income", "label": "Yearly income", "value": 80000, "contribution": -0.7,
                "direction": "decreases risk", "text": "Yearly income of $80,000 - decreases risk"}]
    bar = ui.reasons_figure(reasons).data[0]
    assert list(bar.y) == [reasons[1]["text"], reasons[0]["text"]]          # biggest reason drawn at the top
    assert list(bar.marker.color) == [ui.DECREASES, ui.INCREASES]             # blue decreases, red increases


def test_form_limits_match_the_api():
    """The form must never allow what the API rejects (checked against the API's own schema)."""
    from app.schemas import ApplicantIn
    from clean_data import MAX_AGE, MIN_WORKING_AGE

    props = ApplicantIn.model_json_schema()["properties"]

    def api_range(field):
        p = props[field]
        p = next((x for x in p.get("anyOf", [p]) if x.get("type") != "null"), p)
        lo = p.get("minimum", p.get("exclusiveMinimum", -np.inf))
        return lo, p.get("maximum", np.inf), "exclusiveMinimum" in p

    for field, (lo, hi) in LIMITS.items():
        api_lo, api_hi, exclusive = api_range(field)
        assert lo > api_lo if exclusive else lo >= api_lo, field
        assert hi <= api_hi, field
    assert LIMITS["person_emp_length"][1] == MAX_AGE - MIN_WORKING_AGE        # cross-field rule upper bound
    assert set(DEFAULTS) | {"loan_percent_income"} == set(api.FORM_FIELDS)


# ---------------------------------------------------------------------------
# Error handling of the API client (fake transports, no server)
# ---------------------------------------------------------------------------

def fake_client(status: int, body: dict | None = None) -> api.ApiClient:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json=body or {}))
    return api.ApiClient("token", http=httpx.Client(transport=transport, base_url="http://api"))


def test_api_down_gives_a_friendly_message():
    def refuse(request):
        raise httpx.ConnectError("connection refused")
    client = api.ApiClient(http=httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://api"))
    with pytest.raises(api.ApiError, match="Cannot reach the API"):
        client.me()


@pytest.mark.parametrize("status,message", [(401, "log in again"), (403, "admins only"), (500, "HTTP 500")])
def test_error_messages(status, message):
    with pytest.raises(api.ApiError, match=message):
        fake_client(status).stats()


def test_wrong_login_message_and_missing_drift():
    with pytest.raises(api.ApiError, match="Wrong username or password"):
        fake_client(401).login("x", "y")
    assert fake_client(404).latest_drift() is None


# ---------------------------------------------------------------------------
# The real app, headless (needs the model + the API test database)
# ---------------------------------------------------------------------------

@pytest.fixture
def wired(client, monkeypatch):
    """Make every ApiClient created by the app use the FastAPI TestClient."""
    monkeypatch.setattr(api, "make_http_client", lambda: client)
    return client


def logged_in_app(username: str, password: str) -> AppTest:
    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    at.text_input(key="login_username").input(username)
    at.text_input(key="login_password").input(password)
    at.button[0].click().run()
    return at


def token_for(client, username, password) -> str:
    return client.post("/auth/login", data={"username": username, "password": password}).json()["access_token"]


@needs_model
def test_login_page_and_wrong_password(wired):
    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.endswith("Loan Default Prediction")
    at.text_input(key="login_username").input("test_officer")
    at.text_input(key="login_password").input("wrong")
    at.button[0].click().run()
    assert "Wrong username or password" in at.error[0].value


@needs_model
def test_officer_scores_an_application(wired):
    at = logged_in_app("test_officer", "officer-pass")
    assert not at.exception
    assert at.session_state["role"] == "officer"
    assert at.title[0].value == "New loan application"
    at.text_input(key="applicant_name").input("UI test applicant")
    next(b for b in at.button if b.label == "Score application").click().run()
    assert not at.exception
    result = at.session_state["last_result"]
    assert result["applicant_name"] == "UI test applicant"
    assert result["prediction"]["decision"] in ui.DECISIONS
    assert any("Probability of default" in m.label for m in at.metric)


@needs_model
def test_cross_field_error_is_shown_next_to_the_field(wired):
    at = logged_in_app("test_officer", "officer-pass")
    at.text_input(key="applicant_name").input("Too young to be employed so long")
    at.number_input(key="person_age").set_value(20)
    at.number_input(key="person_emp_length").set_value(10.0)             # 20-year-old employed for 10 years
    next(b for b in at.button if b.label == "Score application").click().run()
    assert "person_emp_length" in at.session_state["form_errors"]
    assert any("Years employed cannot be more than age - 14" in m.value for m in at.markdown)
    assert "Please correct the highlighted fields." in [e.value for e in at.error]


@needs_model
def test_history_page_lists_applications(wired, officer):
    from app.schemas import EXAMPLES
    wired.post("/applications", json=EXAMPLES["high_risk"]["value"], headers=officer)

    def history_page():
        from views import history
        history.page()

    at = AppTest.from_function(history_page, default_timeout=60)
    at.session_state["token"] = token_for(wired, "test_officer", "officer-pass")
    at.run()
    assert not at.exception
    assert at.dataframe and len(at.dataframe[0].value) >= 1
    at.selectbox(key="history_decision").select("REJECT").run()
    assert set(at.dataframe[0].value["Decision"]) == {ui.decision_label("REJECT")}


@needs_model
def test_dashboard_admin_only(wired):
    def dashboard_page():
        from views import dashboard
        dashboard.page()

    at = AppTest.from_function(dashboard_page, default_timeout=60)
    at.session_state["token"] = token_for(wired, "test_admin", "admin-pass")
    at.run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Test ROC-AUC" in labels and "Applications scored" in labels
    assert any("No drift check has run yet" in i.value for i in at.info)

    at.session_state["token"] = token_for(wired, "test_officer", "officer-pass")   # an officer calling the API directly
    at.run()
    assert "This page is for admins only." in [e.value for e in at.error]


@needs_model
def test_empty_name_error_is_shown_under_the_name(wired):
    at = logged_in_app("test_officer", "officer-pass")
    next(b for b in at.button if b.label == "Score application").click().run()
    assert "applicant_name" in at.session_state["form_errors"]
    assert at.session_state["last_result"] is None
