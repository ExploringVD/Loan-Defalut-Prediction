"""Database tables (SQLAlchemy 2.0 ORM). Changes here need an Alembic migration:
    alembic revision --autogenerate -m "what changed"   then   alembic upgrade head
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint("role IN ('admin', 'officer')", name="ck_users_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class LoanApplication(Base):
    """One loan application: the 11 raw fields exactly as entered (grade letter, Y/N) + who entered it."""
    __tablename__ = "loan_applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    applicant_name: Mapped[str] = mapped_column(String(100))
    person_age: Mapped[int] = mapped_column(Integer)
    person_income: Mapped[float] = mapped_column(Float)
    person_home_ownership: Mapped[str] = mapped_column(String(10))
    person_emp_length: Mapped[float | None] = mapped_column(Float, nullable=True)
    loan_intent: Mapped[str] = mapped_column(String(20))
    loan_grade: Mapped[str] = mapped_column(String(1))
    loan_amnt: Mapped[float] = mapped_column(Float)
    loan_int_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    loan_percent_income: Mapped[float] = mapped_column(Float)
    cb_person_default_on_file: Mapped[str] = mapped_column(String(1))
    cb_person_cred_hist_length: Mapped[int] = mapped_column(Integer)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    creator: Mapped["User"] = relationship()
    prediction: Mapped["Prediction | None"] = relationship(back_populates="application", uselist=False)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (CheckConstraint("decision IN ('APPROVE', 'REVIEW', 'REJECT')", name="ck_predictions_decision"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    application_id: Mapped[int] = mapped_column(ForeignKey("loan_applications.id"), unique=True)
    probability: Mapped[float] = mapped_column(Float)
    decision: Mapped[str] = mapped_column(String(10), index=True)
    top_reasons: Mapped[list] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(20))
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)

    application: Mapped["LoanApplication"] = relationship(back_populates="prediction")


class AuditLog(Base):
    """Who did what and when (logins, failed logins, scoring)."""
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50))
    action: Mapped[str] = mapped_column(String(50))
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, index=True)


class DriftReport(Base):
    """One drift check (filled in Step 9)."""
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    batch_name: Mapped[str] = mapped_column(String(50))
    drift_share: Mapped[float] = mapped_column(Float)
    dataset_drift: Mapped[bool] = mapped_column(Boolean)
    prediction_psi: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30))
    report_path: Mapped[str | None] = mapped_column(String(300), nullable=True)
