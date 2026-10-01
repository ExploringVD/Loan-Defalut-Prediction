"""Create the admin and the loan officer from .env (ADMIN_* and OFFICER_*). Safe to run again.
    python -m app.seed
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth import hash_password
from app.config import get_settings
from app.db import SessionLocal
from app.models import User


def ensure_user(db: Session, username: str, password: str, role: str) -> bool:
    """Create the user if it does not exist yet. Returns True if it was created."""
    if not password:
        raise SystemExit(f"No password set for {username!r} - fill in .env first")
    if db.scalar(select(User).where(User.username == username)) is not None:
        return False
    db.add(User(username=username, hashed_password=hash_password(password), role=role))
    db.commit()
    return True


def seed(db: Session) -> None:
    s = get_settings()
    for username, password, role in [(s.admin_username, s.admin_password, "admin"),
                                     (s.officer_username, s.officer_password, "officer")]:
        created = ensure_user(db, username, password, role)
        print(f"{role:8s} {username!r}: {'created' if created else 'already exists'}")


if __name__ == "__main__":
    with SessionLocal() as session:
        seed(session)
