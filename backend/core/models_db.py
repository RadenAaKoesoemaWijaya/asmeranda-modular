"""
SQLAlchemy Database Models for Asmeranda AI.
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, DateTime, Text
try:
    from backend.core.database import Base
except ImportError:
    from core.database import Base


class UserModel(Base):
    """User account entity in database."""
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=True)
    role = Column(String(32), default="analyst", nullable=False)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(String(64), default=lambda: datetime.now(timezone.utc).isoformat(), nullable=False)
    updated_at = Column(String(64), default=lambda: datetime.now(timezone.utc).isoformat(), nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "hashed_password": self.hashed_password,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class AuditLogModel(Base):
    """Persistent Security & Operational Audit Log entity."""
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True, index=True)
    timestamp = Column(String(64), default=lambda: datetime.now(timezone.utc).isoformat(), nullable=False)
    event_type = Column(String(64), index=True, nullable=False)
    user_id = Column(String(64), nullable=True)
    ip_address = Column(String(64), nullable=True)
    status = Column(String(32), nullable=False)
    details = Column(Text, nullable=True)
