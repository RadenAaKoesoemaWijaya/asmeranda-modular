"""
Database Configuration & Session Management for Asmeranda AI.

Provides persistent SQLAlchemy SQLite/PostgreSQL engine and session factory.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

try:
    from backend.core.config import settings
    _DATA_DIR = Path(settings.data_dir)
except Exception:
    _DATA_DIR = Path("data")

_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default DB URL: SQLite in data directory
DB_PATH = _DATA_DIR / "asmeranda.db"
DATABASE_URL = os.getenv("ASMERANDA_DATABASE_URL", f"sqlite:///{DB_PATH.as_posix()}")

# Create SQLAlchemy Engine
# For SQLite, check_same_thread=False is needed for multi-threaded FastAPI execution
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    try:
        from backend.core import models_db  # noqa: F401
    except ImportError:
        try:
            from core import models_db  # noqa: F401
        except ImportError:
            pass
    Base.metadata.create_all(bind=engine)
