"""Database connection and session management."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.api.models import Base
from src.utils.config import get_settings

settings = get_settings()

# Create engine based on config
engine = create_engine(
    settings.database_url,
    echo=False,  # Set to True for SQL query logging during development
    pool_pre_ping=True,  # Verify connections are alive before using them
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Create all tables. Call once at app startup."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for database sessions.

    Usage:
        with get_db() as db:
            user = db.query(User).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_dependency() -> Generator[Session, None, None]:
    """FastAPI dependency for automatic session management.

    Usage in routes:
        @app.get("/users")
        def get_users(db: Session = Depends(get_db_dependency)):
            return db.query(User).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
