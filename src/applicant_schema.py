"""
The rules a loan application must follow - ONE Pydantic schema shared by the API (app/schemas.py re-exports it)
and batch scoring (src/predict_batch.py). Invalid input never reaches the model.
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from clean_data import APPLICANT_FIELDS, MAX_AGE, MIN_WORKING_AGE

HomeOwnership = Literal["RENT", "MORTGAGE", "OWN", "OTHER"]
LoanIntent = Literal["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"]
LoanGrade = Literal["A", "B", "C", "D", "E", "F", "G"]
Decision = Literal["APPROVE", "REVIEW", "REJECT"]

MAX_PERCENT_INCOME_GAP = 0.10   # loan_percent_income must be close to loan_amnt / person_income


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
