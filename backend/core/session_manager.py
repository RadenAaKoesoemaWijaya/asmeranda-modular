"""
Session Management Module.

Provides secure session management with:
- Session creation with unique IDs
- Session expiration
- Session validation
- Session storage and cleanup
"""
from __future__ import annotations

import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import secrets

logger = logging.getLogger("asmeranda.security.session")


class SessionManager:
    """Manages user sessions with security features."""
    
    def __init__(self, session_timeout_minutes: int = 30, storage_dir: Optional[Path] = None):
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.storage_dir = storage_dir or Path("data/sessions")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._load_sessions()
    
    def _load_sessions(self):
        """Load existing sessions from disk using secure JSON."""
        try:
            session_file = self.storage_dir / "sessions.json"
            if session_file.exists():
                with open(session_file, "r", encoding="utf-8") as f:
                    raw_sessions = json.load(f)
                    for sid, sdata in raw_sessions.items():
                        # Parse ISO datetimes
                        if "created_at" in sdata and isinstance(sdata["created_at"], str):
                            sdata["created_at"] = datetime.fromisoformat(sdata["created_at"])
                        if "expires_at" in sdata and isinstance(sdata["expires_at"], str):
                            sdata["expires_at"] = datetime.fromisoformat(sdata["expires_at"])
                        if "last_activity" in sdata and isinstance(sdata["last_activity"], str):
                            sdata["last_activity"] = datetime.fromisoformat(sdata["last_activity"])
                        self._sessions[sid] = sdata
                logger.info(f"Loaded {len(self._sessions)} sessions from disk")
        except Exception as e:
            logger.warning(f"Failed to load sessions: {e}")
            self._sessions = {}
    
    def _save_sessions(self):
        """Save sessions to disk using secure JSON."""
        try:
            session_file = self.storage_dir / "sessions.json"
            serializable = {}
            for sid, sdata in self._sessions.items():
                copy_s = dict(sdata)
                for dt_key in ["created_at", "expires_at", "last_activity"]:
                    if isinstance(copy_s.get(dt_key), datetime):
                        copy_s[dt_key] = copy_s[dt_key].isoformat()
                serializable[sid] = copy_s
                
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new session.
        
        Parameters
        ----------
        user_id : str
            User identifier
        metadata : dict
            Additional session metadata
            
        Returns
        -------
        str
            Session ID
        """
        session_id = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "expires_at": now + self.session_timeout,
            "last_activity": now,
            "metadata": metadata or {},
            "ip_address": None,
            "user_agent": None
        }
        
        self._sessions[session_id] = session_data
        self._save_sessions()
        
        logger.info(
            f"Session created: {session_id[:8]}... for user: {user_id or 'anonymous'}"
        )
        
        return session_id
    
    def validate_session(
        self,
        session_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Validate a session.
        
        Parameters
        ----------
        session_id : str
            Session ID to validate
        ip_address : str
            Client IP address for additional validation
        user_agent : str
            User agent string for additional validation
            
        Returns
        -------
        bool
            True if session is valid
        """
        if session_id not in self._sessions:
            logger.warning(f"Session not found: {session_id[:8]}...")
            return False
        
        session = self._sessions[session_id]
        now = datetime.now(timezone.utc)
        
        # Check expiration
        if now > session["expires_at"]:
            logger.warning(f"Session expired: {session_id[:8]}...")
            self.delete_session(session_id)
            return False
        
        # Check IP address if provided and session has one
        if ip_address and session.get("ip_address"):
            if ip_address != session["ip_address"]:
                logger.warning(f"IP address mismatch for session: {session_id[:8]}...")
                return False
        
        # Check user agent if provided and session has one
        if user_agent and session.get("user_agent"):
            if user_agent != session["user_agent"]:
                logger.warning(f"User agent mismatch for session: {session_id[:8]}...")
                return False
        
        # Update last activity
        session["last_activity"] = now
        self._save_sessions()
        
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Parameters
        ----------
        session_id : str
            Session ID to delete
            
        Returns
        -------
        bool
            True if session was deleted
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_sessions()
            logger.info(f"Session deleted: {session_id[:8]}...")
            return True
        return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.
        
        Returns
        -------
        int
            Number of sessions cleaned up
        """
        now = datetime.now(timezone.utc)
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if now > session["expires_at"]
        ]
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
        
        if expired_sessions:
            self._save_sessions()
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information without validating.
        
        Parameters
        ----------
        session_id : str
            Session ID
            
        Returns
        -------
        dict or None
            Session information if exists
        """
        return self._sessions.get(session_id)
    
    def extend_session(self, session_id: str, additional_minutes: int = 30) -> bool:
        """
        Extend session expiration.
        
        Parameters
        ----------
        session_id : str
            Session ID to extend
        additional_minutes : int
            Additional minutes to add to expiration
            
        Returns
        -------
        bool
            True if session was extended
        """
        if session_id not in self._sessions:
            return False
        
        session = self._sessions[session_id]
        session["expires_at"] = session["expires_at"] + timedelta(minutes=additional_minutes)
        session["last_activity"] = datetime.now(timezone.utc)
        self._save_sessions()
        
        logger.info(f"Session extended: {session_id[:8]}... by {additional_minutes} minutes")
        return True


# Global session manager instance
session_manager = SessionManager()