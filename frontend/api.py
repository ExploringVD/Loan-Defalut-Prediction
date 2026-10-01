"""Everything the frontend knows about the backend: it ONLY talks to the FastAPI API (never to the model or the
database). API problems become ApiError with a message that can be shown to the user."""
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env", override=False)
API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT_SECONDS = 20

# Field names of the application form (same as the API / the Kaggle file)
FORM_FIELDS = [
    "applicant_name", "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length",
]


# How each field is called on the form (used to make API messages readable)
FIELD_LABELS = {
    "applicant_name": "Applicant name", "person_age": "age", "person_income": "yearly income",
    "person_home_ownership": "Home ownership", "person_emp_length": "Years employed", "loan_intent": "Purpose",
    "loan_grade": "Loan grade", "loan_amnt": "loan amount", "loan_int_rate": "Interest rate",
    "loan_percent_income": "Loan as % of income", "cb_person_default_on_file": "Past default",
    "cb_person_cred_hist_length": "Credit history length",
}


def readable(message: str) -> str:
    """'person_emp_length cannot be more than person_age - 14' -> 'Years employed cannot be more than age - 14'."""
    for field in sorted(FIELD_LABELS, key=len, reverse=True):     # longest first: loan_amnt before loan_...
        message = message.replace(field, FIELD_LABELS[field])
    return message[:1].upper() + message[1:]


class ApiError(Exception):
    """A problem talking to the API, with a message for the user and (for 422) errors per form field."""

    def __init__(self, message: str, status: int | None = None, field_errors: dict | None = None,
                 general_errors: list | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.field_errors = field_errors or {}
        self.general_errors = general_errors or []


def make_http_client() -> httpx.Client:
    """The HTTP client used for every call (the tests replace it with FastAPI's TestClient)."""
    return httpx.Client(base_url=API_URL, timeout=TIMEOUT_SECONDS)


def parse_validation_errors(detail) -> tuple[dict, list]:
    """FastAPI 422 'detail' -> ({field: message}, [messages that belong to no single field]).

    Field checks have the field name in 'loc'. Checks across fields (e.g. employment length vs age) have
    loc = ['body'], so the first field named in the message is used."""
    field_errors, general = {}, []
    for err in detail if isinstance(detail, list) else []:
        loc = [str(x) for x in err.get("loc", [])]
        msg = str(err.get("msg", "Invalid value")).removeprefix("Value error, ")
        mentioned = sorted((msg.find(f), f) for f in FORM_FIELDS if f in msg)   # the field named first in the message
        field = loc[-1] if loc and loc[-1] in FORM_FIELDS else (mentioned[0][1] if mentioned else None)
        if field:
            field_errors.setdefault(field, readable(msg))
        else:
            general.append(readable(msg))
    return field_errors, general


class ApiClient:
    def __init__(self, token: str | None = None, http=None):
        self.token = token
        self.http = http or make_http_client()

    def _request(self, method: str, path: str, **kwargs):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            r = self.http.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError:
            raise ApiError(f"Cannot reach the API at {API_URL}. Is it running? Start it with: uvicorn app.main:app")
        if r.status_code < 400:
            return r.json()
        if r.status_code == 401:
            raise ApiError("Your session has expired - please log in again.", 401)
        if r.status_code == 403:
            raise ApiError("This page is for admins only.", 403)
        if r.status_code == 404:
            raise ApiError("Not found.", 404)
        if r.status_code == 422:
            fields, general = parse_validation_errors(r.json().get("detail"))
            raise ApiError("Please correct the highlighted fields.", 422, fields, general)
        raise ApiError(f"The API returned an error (HTTP {r.status_code}). Please try again.", r.status_code)

    def login(self, username: str, password: str) -> dict:
        """-> {"access_token", "role", ...}. Wrong credentials give a clear message instead of 'session expired'."""
        try:
            return self._request("POST", "/auth/login", data={"username": username, "password": password})
        except ApiError as e:
            if e.status in (401, 422):
                raise ApiError("Wrong username or password.", e.status)
            raise

    def me(self) -> dict:
        return self._request("GET", "/auth/me")

    def create_application(self, body: dict) -> dict:
        return self._request("POST", "/applications", json=body)

    def list_applications(self, page: int = 1, page_size: int = 50, decision: str | None = None,
                          date_from=None, date_to=None) -> dict:
        params = {"page": page, "page_size": page_size, "decision": decision,
                  "date_from": date_from.isoformat() if date_from else None,
                  "date_to": date_to.isoformat() if date_to else None}
        return self._request("GET", "/applications", params={k: v for k, v in params.items() if v is not None})

    def get_application(self, application_id: int) -> dict:
        return self._request("GET", f"/applications/{application_id}")

    def model_info(self) -> dict:
        return self._request("GET", "/model/info")

    def stats(self) -> dict:
        return self._request("GET", "/monitoring/stats")

    def run_drift(self, batch: str) -> dict:
        """Run a drift check now: batch = 'live', 'drift' or 'no_drift' (admin only, takes a few seconds)."""
        return self._request("POST", "/monitoring/drift/run", params={"batch": batch})

    def latest_drift(self) -> dict | None:
        """Latest drift check, or None if there is none yet (the drift endpoints arrive in Step 9)."""
        try:
            return self._request("GET", "/monitoring/drift/latest")
        except ApiError as e:
            if e.status == 404:
                return None
            raise
