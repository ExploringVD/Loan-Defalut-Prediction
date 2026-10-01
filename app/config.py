"""Settings read from the .env file in the project folder (see .env.example).

Real environment variables win over .env, so tests and Docker can override any value.
"""
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from app import PROJECT_DIR

load_dotenv(PROJECT_DIR / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    database_url: str
    jwt_secret: str
    jwt_expire_minutes: int
    admin_username: str
    admin_password: str
    officer_username: str
    officer_password: str
    jwt_algorithm: str = "HS256"


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set - copy .env.example to .env and fill it in")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        database_url=_required("DATABASE_URL"),
        jwt_secret=_required("JWT_SECRET"),
        jwt_expire_minutes=int(os.getenv("JWT_EXPIRE_MINUTES", "60")),
        admin_username=os.getenv("ADMIN_USERNAME", "admin"),
        admin_password=os.getenv("ADMIN_PASSWORD", ""),
        officer_username=os.getenv("OFFICER_USERNAME", "officer"),
        officer_password=os.getenv("OFFICER_PASSWORD", ""),
    )
