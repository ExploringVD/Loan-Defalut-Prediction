"""Request and response shapes (Pydantic). Invalid input never reaches the model: FastAPI answers 422."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from applicant_schema import ApplicantIn, Decision  # noqa: F401 - rules live in src/, shared with batch scoring
from explain import EXAMPLE_APPLICANT

# Swagger examples: two real loans from the data
EXAMPLES = {
    "low_risk": {
        "summary": "Low risk - real repaid loan (loan_id 6560)",
        "value": {"applicant_name": "Example low risk", **EXAMPLE_APPLICANT},
    },
    "high_risk": {
        "summary": "High risk - real defaulted loan (loan_id 32106)",
        "value": {"applicant_name": "Example high risk", "person_age": 38, "person_income": 18720,
                  "person_home_ownership": "RENT", "person_emp_length": 0, "loan_intent": "DEBTCONSOLIDATION",
                  "loan_grade": "D", "loan_amnt": 6000, "loan_int_rate": 15.99, "loan_percent_income": 0.32,
                  "cb_person_default_on_file": "Y", "cb_person_cred_hist_length": 12},
    },
    "missing_optional": {
        "summary": "Interest rate and employment length not given (the model fills them in)",
        "value": {"applicant_name": "Example missing values", **EXAMPLE_APPLICANT,
                  "loan_int_rate": None, "person_emp_length": None},
    },
}


class ReasonOut(BaseModel):
    feature: str
    label: str
    value: str | float | int | None
    contribution: float = Field(description="SHAP value in log-odds: > 0 increases risk")
    direction: Literal["increases risk", "decreases risk"]
    text: str


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    probability: float = Field(description="Calibrated probability of default (0-1)")
    decision: Decision
    top_reasons: list[ReasonOut]
    model_version: str
    latency_ms: float = Field(description="Time to score + explain, in milliseconds")
    created_at: datetime


class ApplicationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    applicant_name: str
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float | None
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float | None
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int
    created_by: str
    created_at: datetime
    prediction: PredictionOut | None


class ApplicationPage(BaseModel):
    items: list[ApplicationOut]
    total: int
    page: int
    page_size: int


class BatchRowOut(BaseModel):
    """One scored row of an uploaded CSV (not stored in the database)."""
    row: int = Field(description="Row number in the file (1 = first row under the header)")
    loan_id: int | str | None = None
    applicant_name: str | None = None
    probability: float
    decision: Decision
    top_reason: str
    loan_status: int | None = None


class BatchRejectedOut(BaseModel):
    """A row that broke the validation rules - not scored."""
    row: int
    loan_id: int | str | None = None
    applicant_name: str | None = None
    reason: str


class BatchEvaluationOut(BaseModel):
    """Only when the CSV has loan_status (the real outcome)."""
    rows_with_loan_status: int
    roc_auc: float | None
    default_rate_by_band: dict[str, float | None]
    rows_by_band: dict[str, int]


class BatchSummaryOut(BaseModel):
    rows_total: int
    rows_scored: int
    rows_rejected: int
    decisions: dict[str, int]
    cutoffs: dict[str, float]
    model_version: str
    seconds: float
    evaluation: BatchEvaluationOut | None = None


class BatchOut(BaseModel):
    scored: list[BatchRowOut]
    rejected: list[BatchRejectedOut]
    summary: BatchSummaryOut


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    expires_in_minutes: int


class UserOut(BaseModel):
    username: str
    role: str


class DayStats(BaseModel):
    date: str
    requests: int
    approve: int
    review: int
    reject: int


class DriftReportOut(BaseModel):
    """Latest drift check (what the Dashboard shows)."""
    model_config = ConfigDict(from_attributes=True)

    created_at: datetime
    batch_name: str
    drift_share: float = Field(description="Share of the 11 features that drifted")
    dataset_drift: bool
    prediction_psi: float = Field(description="PSI of the predicted probability")
    status: Literal["DRIFT DETECTED", "NO DRIFT"]
    report_path: str | None


class DriftRunOut(BaseModel):
    """Result of POST /monitoring/drift/run (status SKIPPED when live mode has too few applications)."""
    status: Literal["DRIFT DETECTED", "NO DRIFT", "SKIPPED"]
    batch_name: str
    message: str | None = None
    n_current: int
    created_at: datetime | None = None
    drift_share: float | None = None
    n_drifted: int | None = None
    drifted_features: list[str] = []
    dataset_drift: bool | None = None
    prediction_psi: float | None = None
    mean_probability: dict[str, float] | None = None
    report_path: str | None = None


class StatsOut(BaseModel):
    total_requests: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    decisions: dict[str, int]
    last_7_days: list[DayStats]
