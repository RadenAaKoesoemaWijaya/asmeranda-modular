"""
Centralized error handling untuk Asmeranda.

Menyediakan:
- ``ErrorHandler`` - logika handling error (classify, format, log, saran).
- ``DisplayCallback`` - hook opsional untuk menampilkan error ke UI
  (Streamlit, console, WebSocket, dll). Default: logging saja.
- ``format_error_info(...)`` - helper untuk membangun struktur error
  tanpa efek UI.

Backward compatible: pemanggil lama yang mengharapkan
``handler.handle_error(err)`` tetap jalan - default display callback
memanggil ``st.error/expander/code`` bila Streamlit aktif.
"""
from __future__ import annotations

import logging
import os
import traceback
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.log import configure_logging, get_logger
from core.state import get_state


# ---------------------------------------------------------------------------
# Optional streamlit
# ---------------------------------------------------------------------------
try:
    import streamlit as _st  # type: ignore

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    _st = None
    _ST_AVAILABLE = False


# Tipe callback display: (error_info: dict) -> None
DisplayCallback = Callable[[Dict[str, Any]], None]


def _default_streamlit_display(error_info: Dict[str, Any], language: str = "id") -> None:
    """Display error via Streamlit (legacy UI)."""
    if not _ST_AVAILABLE:
        return
    try:
        _st.error(f"❌ {error_info['message']}")
        if error_info.get("suggestions"):
            label = "💡 Saran Perbaikan" if language == "id" else "💡 Suggestions"
            with _st.expander(label):
                for s in error_info["suggestions"]:
                    _st.write(f"• {s}")
        # Detail teknis
        checkbox_key = (
            f"tech_details_{error_info.get('context', '')}_"
            f"{hash(error_info.get('technical_details', ''))}"
        )
        label = "Tampilkan detail teknis" if language == "id" else "Show technical details"
        if _st.checkbox(label, key=checkbox_key):
            _st.code(str(error_info.get("technical_details", "")))
            _st.caption(f"Context: {error_info.get('context', '')}")
            _st.caption(f"Time: {error_info.get('timestamp', '')}")
    except Exception:
        # Streamlit mungkin tidak aktif di thread ini
        pass


def _silent_display(error_info: Dict[str, Any]) -> None:
    """Display no-op (untuk backend FastAPI)."""
    return None


# ---------------------------------------------------------------------------
# ErrorHandler
# ---------------------------------------------------------------------------
class ErrorHandler:
    """Centralized error handling with user-friendly messages."""

    def __init__(
        self,
        language: str = "id",
        auth_db: Any = None,
        display_callback: Optional[DisplayCallback] = None,
    ):
        self.language = language
        self.auth_db = auth_db
        self.error_messages = self._load_error_messages()
        # Inisialisasi logging via core.log agar idempotent
        configure_logging(force=False)
        self.logger = get_logger("asmeranda.error_handler")

        # Tentukan display callback default
        if display_callback is not None:
            self._display = display_callback
        else:
            # Default: jika di lingkungan Streamlit, gunakan st.error/expander;
            # jika tidak, gunakan no-op (logging saja).
            if _ST_AVAILABLE:
                self._display = lambda info: _default_streamlit_display(info, self.language)
            else:
                self._display = _silent_display

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_display_callback(self, callback: DisplayCallback) -> None:
        """Ganti display callback (untuk FastAPI WebSocket, dsb)."""
        self._display = callback

    def set_language(self, language: str) -> None:
        self.language = language

    # ------------------------------------------------------------------
    # Error messages
    # ------------------------------------------------------------------
    def _load_error_messages(self) -> Dict[str, Dict[str, str]]:
        return {
            "id": {
                "data_empty": "Dataset kosong atau tidak valid. Silakan unggah dataset yang valid.",
                "column_not_found": "Kolom '{column}' tidak ditemukan dalam dataset.",
                "model_training_failed": "Pelatihan model gagal: {error}. Silakan periksa parameter atau data.",
                "prediction_failed": "Prediksi gagal: {error}. Pastikan model telah dilatih dengan benar.",
                "invalid_parameter": "Parameter '{param}' tidak valid: {value}",
                "memory_error": "Memori tidak cukup untuk operasi ini. Coba dengan dataset yang lebih kecil.",
                "timeout_error": "Operasi terlalu lama. Coba dengan parameter yang lebih sederhana.",
                "unknown_error": "Terjadi kesalahan tidak terduga: {error}. Silakan coba lagi atau hubungi admin.",
            },
            "en": {
                "data_empty": "Dataset is empty or invalid. Please upload a valid dataset.",
                "column_not_found": "Column '{column}' not found in dataset.",
                "model_training_failed": "Model training failed: {error}. Please check parameters or data.",
                "prediction_failed": "Prediction failed: {error}. Ensure model is properly trained.",
                "invalid_parameter": "Parameter '{param}' is invalid: {value}",
                "memory_error": "Insufficient memory for this operation. Try with a smaller dataset.",
                "timeout_error": "Operation took too long. Try with simpler parameters.",
                "unknown_error": "Unexpected error occurred: {error}. Please try again or contact admin.",
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def handle_error(
        self,
        error: Exception,
        context: str = "",
        user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Handle errors with proper logging and user messages."""
        self.logger.error(f"Error in {context}: {error}")

        # Log to database if available - tidak boleh crash kalau gagal
        if self.auth_db is not None:
            try:
                username = "system"
                try:
                    state = get_state()
                    username = state.get("current_username") or "system"
                except Exception:
                    pass
                try:
                    self.auth_db.record_activity(
                        username,
                        f"ERROR: {context}",
                        metadata=str(error)[:500],
                    )
                except Exception as log_err:
                    self.logger.warning(f"Failed to log error to database: {log_err}")
            except Exception as outer_err:
                self.logger.warning(f"DB logging skipped: {outer_err}")

        error_type = self._classify_error(error)
        if user_message is None:
            user_message = self._get_user_message(error_type, str(error))

        error_info: Dict[str, Any] = {
            "type": error_type,
            "message": user_message,
            "technical_details": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "timestamp": pd.Timestamp.now(),
            "suggestions": self._get_suggestions(error_type),
        }

        # Panggil display callback (default = streamlit atau silent)
        try:
            self._display(error_info)
        except Exception as disp_err:
            self.logger.warning(f"Display callback failed: {disp_err}")

        return error_info

    # ------------------------------------------------------------------
    # Classify & message
    # ------------------------------------------------------------------
    def _classify_error(self, error: Exception) -> str:
        error_str = str(error).lower()
        if isinstance(error, MemoryError):
            return "memory_error"
        if isinstance(error, KeyError):
            return "column_not_found"
        if isinstance(error, ValueError):
            return "invalid_parameter"
        if isinstance(error, TimeoutError):
            return "timeout_error"
        if "memory" in error_str or "ram" in error_str:
            return "memory_error"
        if "timeout" in error_str or "time" in error_str:
            return "timeout_error"
        if "column" in error_str or "key" in error_str:
            return "column_not_found"
        if "empty" in error_str or "none" in error_str:
            return "data_empty"
        if "parameter" in error_str or "argument" in error_str:
            return "invalid_parameter"
        if "model" in error_str or "training" in error_str:
            return "model_training_failed"
        if "prediction" in error_str or "predict" in error_str:
            return "prediction_failed"
        return "unknown_error"

    def _get_user_message(self, error_type: str, error_details: str) -> str:
        messages = self.error_messages.get(self.language, self.error_messages["en"])
        template = messages.get(error_type, messages["unknown_error"])
        try:
            return template.format(error=error_details)
        except Exception:
            return template

    def _get_suggestions(self, error_type: str) -> List[str]:
        suggestions: Dict[str, List[str]] = {
            "data_empty": [
                "Upload a valid dataset",
                "Check if the file was uploaded correctly",
                "Verify the file format (CSV, Excel, etc.)",
            ],
            "column_not_found": [
                "Check column names for typos",
                "Verify the column exists in your dataset",
                "Refresh the dataset if you made changes",
            ],
            "model_training_failed": [
                "Check your data quality and preprocessing",
                "Verify model parameters are appropriate",
                "Try a different model type",
                "Ensure you have enough training data",
            ],
            "prediction_failed": [
                "Ensure your model is trained successfully",
                "Check if input data matches training data format",
                "Verify model compatibility with prediction task",
            ],
            "memory_error": [
                "Use a smaller dataset or sample",
                "Reduce model complexity",
                "Close other applications to free memory",
                "Consider cloud deployment for large datasets",
            ],
            "timeout_error": [
                "Reduce dataset size",
                "Use simpler model parameters",
                "Disable complex features like cross-validation",
                "Use fewer features for training",
            ],
        }
        return suggestions.get(error_type, ["Please check your input data and parameters"])

    # ------------------------------------------------------------------
    # Validators (pure)
    # ------------------------------------------------------------------
    def validate_input(
        self,
        value: Any,
        expected_type: type,
        param_name: str,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
    ) -> Dict[str, Any]:
        try:
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    return {
                        "valid": False,
                        "error": f"Parameter '{param_name}' must be of type {expected_type.__name__}",
                        "value": value,
                    }
            if min_value is not None and value < min_value:
                return {
                    "valid": False,
                    "error": f"Parameter '{param_name}' must be >= {min_value}",
                    "value": value,
                }
            if max_value is not None and value > max_value:
                return {
                    "valid": False,
                    "error": f"Parameter '{param_name}' must be <= {max_value}",
                    "value": value,
                }
            return {"valid": True, "value": value, "error": None}
        except Exception as e:
            return {
                "valid": False,
                "error": f"Validation failed for parameter '{param_name}': {e}",
                "value": value,
            }
