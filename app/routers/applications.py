"""Score new loan applications and look at past ones."""
from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.auth import get_current_user
from app.db import get_db
from app.models import AuditLog, LoanApplication, Prediction, User
from app.schemas import EXAMPLES, ApplicantIn, ApplicationOut, ApplicationPage, Decision
from app.services.scoring import score

router = APIRouter(prefix="/applications", tags=["applications"])


def to_out(application: LoanApplication) -> ApplicationOut:
    """Database row -> response (created_by is shown as the username, not the user id)."""
    data = {c.name: getattr(application, c.name) for c in LoanApplication.__table__.columns}
    data["created_by"] = application.creator.username
    data["prediction"] = application.prediction
    return ApplicationOut.model_validate(data)


@router.post("", response_model=ApplicationOut, status_code=status.HTTP_201_CREATED)
def create_application(
    applicant: Annotated[ApplicantIn, Body(openapi_examples=EXAMPLES)],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Validate the application, score it (calibrated probability + APPROVE / REVIEW / REJECT + top 5 SHAP
    reasons), and save the application, the prediction and an audit row."""
    result = score(applicant.applicant_fields())
    application = LoanApplication(**applicant.model_dump(), created_by=user.id)
    application.prediction = Prediction(**result)
    db.add(application)
    db.flush()   # gives the application its id for the audit row
    db.add(AuditLog(username=user.username, action="score_application",
                    details={"application_id": application.id, "decision": result["decision"],
                             "probability": result["probability"]}))
    db.commit()
    return to_out(application)


@router.get("", response_model=ApplicationPage)
def list_applications(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    decision: Decision | None = Query(None, description="Only APPROVE, REVIEW or REJECT"),
    date_from: date | None = Query(None, description="Created on or after this day (UTC), e.g. 2026-10-01"),
    date_to: date | None = Query(None, description="Created on or before this day (UTC)"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Past applications, newest first."""
    query = select(LoanApplication).options(selectinload(LoanApplication.prediction), selectinload(LoanApplication.creator))
    count = select(func.count(LoanApplication.id))
    conditions = []
    if date_from:
        conditions.append(LoanApplication.created_at >= datetime.combine(date_from, time.min, tzinfo=timezone.utc))
    if date_to:   # whole day included: everything before the next midnight
        conditions.append(LoanApplication.created_at < datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc))
    if decision:
        query, count = query.join(Prediction), count.join(Prediction)
        conditions.append(Prediction.decision == decision)
    if conditions:
        query, count = query.where(*conditions), count.where(*conditions)
    rows = db.scalars(query.order_by(LoanApplication.id.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return ApplicationPage(items=[to_out(a) for a in rows], total=db.scalar(count), page=page, page_size=page_size)


@router.get("/{application_id}", response_model=ApplicationOut)
def get_application(application_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    application = db.get(LoanApplication, application_id)
    if application is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Application {application_id} not found")
    return to_out(application)
