"""Request and response shapes (Pydantic). Invalid input never reaches the model: FastAPI answers 422."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clean_data import APPLICANT_FIELDS, MAX_AGE, MIN_WORKING_AGE
from explain import EXAMPLE_APPLICANT

HomeOwnership = Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
LoanIntent = Literal["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
LoanGrade = Literal["A", "B", "C", "D", "E", "F", "G"]
Decision = Literal["APPROVE", "REVIEW", "REJECT"]

MAX_PERCENT_INCOME_GAP = 0.10   # loan_percent_income must be close to loan_amnt / person_income

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


class ApplicantIn(BaseModel):
    """A loan application: applicant name + the 11 raw fields of the Kaggle file."""
    model_config = ConfigDict(extra="forbid")

    applicant_name: str = Field(min_length=1, max_length=100)
    person_age: int = Field(ge=18, le=MAX_AGE, description="Age in years")
    person_income: float = Field(gt=0, le=10_000_000, description="Yearly income in $")
    person_home_ownership: HomeOwnership
    person_emp_length: float | None = Field(default=None, ge=0, description="Years employed (optional)")
    loan_intent: LoanIntent
    loan_grade: LoanGrade = Field(description="Lender's risk grade, A (safest) to G")
    loan_amnt: float = Field(gt=0, le=100_000, description="Loan amount in $")
    loan_int_rate: float | None = Field(default=None, gt=0, le=40, description="Interest rate in % (optional)")
    loan_percent_income: float = Field(ge=0, le=1, description="Loan amount / yearly income, e.g. 0.25")
    cb_person_default_on_file: Literal["Y", "N"] = Field(description="Defaulted before? (credit bureau)")
    cb_person_cred_hist_length: int = Field(ge=0, description="Credit history length in years")

    @field_validator("person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file", mode="before")
    @classmethod
    def upper_case(cls, value):
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def fields_agree(self):
        if self.person_emp_length is not None and self.person_emp_length > self.person_age - MIN_WORKING_AGE:
            raise ValueError(f"person_emp_length cannot be more than person_age - {MIN_WORKING_AGE}")
        if self.cb_person_cred_hist_length > self.person_age:
            raise ValueError("cb_person_cred_hist_length cannot be more than person_age")
        ratio = self.loan_amnt / self.person_income
        if abs(self.loan_percent_income - ratio) > MAX_PERCENT_INCOME_GAP:
            raise ValueError(f"loan_percent_income ({self.loan_percent_income}) does not match "
                             f"loan_amnt / person_income ({ratio:.2f})")
        return self

    def applicant_fields(self) -> dict:
        """The 11 raw fields only (what the model needs)."""
        return self.model_dump(include=set(APPLICANT_FIELDS))


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


class StatsOut(BaseModel):
    total_requests: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    decisions: dict[str, int]
    last_7_days: list[DayStats]
