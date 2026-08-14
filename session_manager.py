"""
Session / Workflow state manager untuk Asmeranda.

Inti logikanya murni-Python dan menerima ``state`` (dict) sebagai
dependency. Bila ``state`` tidak diberikan secara eksplisit, modul
akan otomatis memakai ``st.session_state`` (backward compatibility
dengan legacy UI Streamlit).

API publik: ``SessionStateManager()`` (alias ``SessionManager``)
masih bisa dipanggil seperti sebelumnya.
"""
from __future__ import annotations

import pickle
from typing import Any, Dict, Optional

import pandas as pd

from core.state import DEFAULT_KEYS, get_state

# ---------------------------------------------------------------------------
# Optional streamlit - hanya untuk backward compatibility
# ---------------------------------------------------------------------------
try:
    import streamlit as _st  # type: ignore

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    _st = None
    _ST_AVAILABLE = False


class SessionStateManager:
    """Centralized session state management with validation and persistence."""

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        self.required_keys = dict(DEFAULT_KEYS)
        # Backward-compat: jika tidak ada state, pakai st.session_state
        if state is None:
            self.state: Dict[str, Any] = get_state()
        else:
            self.state = state
        self.initialize_session_state()

    # ------------------------------------------------------------------
    # Initialization & reset
    # ------------------------------------------------------------------
    def initialize_session_state(self) -> None:
        """Initialize all required session state variables with validation."""
        for key, default_value in self.required_keys.items():
            if key not in self.state:
                self.state[key] = default_value

    def reset(self) -> None:
        """Reset workflow keys (mempertahankan key global: language, dll)."""
        for key, default in self.required_keys.items():
            self.state[key] = default

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_data_flow(self, from_tab: str, to_tab: str) -> Dict[str, Any]:
        """Validate data availability between workflow steps."""
        validation_rules = {
            ("upload", "eda"): ["data"],
            ("eda", "preprocessing"): ["data", "numerical_columns", "categorical_columns"],
            ("preprocessing", "training"): ["X_train", "X_test", "y_train", "y_test", "problem_type"],
            ("training", "interpretation"): ["model_results"],
        }

        missing: list[str] = []
        required_keys = validation_rules.get((from_tab, to_tab), [])

        for key in required_keys:
            value = self.state.get(key)
            if value is None:
                if key in ("numerical_columns", "categorical_columns") and not value:
                    missing.append(key)
                elif key not in ("numerical_columns", "categorical_columns"):
                    missing.append(key)

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "message": f"Missing required data: {', '.join(missing)}" if missing else "Valid",
        }

    # ------------------------------------------------------------------
    # Snapshot / persistence
    # ------------------------------------------------------------------
    def save_workflow_state(self, tab_name: str) -> None:
        """Save current workflow state for recovery."""
        state_snapshot: Dict[str, Any] = {}
        for key in self.required_keys:
            if key in self.state:
                # Handle non-serializable objects
                try:
                    pickle.dumps(self.state[key])
                    state_snapshot[key] = self.state[key]
                except (pickle.PicklingError, TypeError):
                    if hasattr(self.state[key], "to_dict"):
                        state_snapshot[key] = self.state[key].to_dict()
                    else:
                        state_snapshot[key] = str(self.state[key])

        self.state[f"workflow_state_{tab_name}"] = state_snapshot

    def clear_workflow_data(self, keep_basic: bool = True) -> None:
        """Clear workflow data while preserving basic settings."""
        basic_keys = {"language", "authenticated", "current_username", "user_email"}

        for key in self.required_keys:
            if keep_basic and key in basic_keys:
                continue
            self.state[key] = self.required_keys[key]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def initialize_data_session(self, data: pd.DataFrame) -> None:
        """Initialize session with data context (compatibility method)."""
        self.state["data"] = data

    # ------------------------------------------------------------------
    # Streamlit-specific helpers (hanya relevan di legacy UI)
    # ------------------------------------------------------------------
    def ensure_streamlit_initialized(self) -> None:
        """Panggil setelah ``st.set_page_config`` agar key global ter-isi."""
        if not _ST_AVAILABLE:
            return
        if "language" not in _st.session_state:  # type: ignore[union-attr]
            _st.session_state["language"] = "id"
        if "authenticated" not in _st.session_state:
            _st.session_state["authenticated"] = False
        if "current_username" not in _st.session_state:
            _st.session_state["current_username"] = ""


# Backward-compatibility alias
SessionManager = SessionStateManager
