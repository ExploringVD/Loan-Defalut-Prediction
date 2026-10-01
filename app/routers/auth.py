"""POST /auth/login -> JWT token; GET /auth/me -> who am I."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.auth import authenticate, create_access_token, get_current_user
from app.config import get_settings
from app.db import get_db
from app.models import AuditLog, User
from app.schemas import TokenOut, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Log in with username + password (form fields). Use the token as `Authorization: Bearer <token>`."""
    user = authenticate(db, form.username, form.password)
    db.add(AuditLog(username=form.username, action="login" if user else "login_failed"))
    db.commit()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    return TokenOut(access_token=create_access_token(user), role=user.role,
                    expires_in_minutes=get_settings().jwt_expire_minutes)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut(username=user.username, role=user.role)
