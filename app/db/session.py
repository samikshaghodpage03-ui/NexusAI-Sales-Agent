from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.config import get_settings
from app.db.base import Base
# Import models so they register on Base.metadata
from app.db import models  # noqa: F401

settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()