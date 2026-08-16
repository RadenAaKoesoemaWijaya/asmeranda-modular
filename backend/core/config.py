"""
Konfigurasi terpusat untuk Asmeranda Backend.

Semua nilai konfigurasi dibaca dari environment variables (dengan
prefix ``ASMERANDA_``). Tipe-typed via pydantic-settings (atau
pydantic v2 BaseSettings fallback) agar IDE bisa autocomplete.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

# Pakai pydantic-settings bila tersedia, kalau tidak fallback ke dataclass
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore

    _HAVE_PYDANTIC_SETTINGS = True
except Exception:  # pragma: no cover
    _HAVE_PYDANTIC_SETTINGS = False

try:
    from pydantic import Field
except Exception:  # pragma: no cover
    Field = None  # type: ignore


# ---------------------------------------------------------------------------
# Path absolut root project (folder berisi ``core/``, ``ml_engine/``, dll)
# Ini agar ``backend`` bisa hidup di sub-folder tanpa khawatir
# ``sys.path`` resolution.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def _build_settings():
    if _HAVE_PYDANTIC_SETTINGS:

        class Settings(BaseSettings):
            model_config = SettingsConfigDict(
                env_prefix="ASMERANDA_",
                env_file=str(PROJECT_ROOT / ".env"),
                env_file_encoding="utf-8",
                extra="ignore",
            )

            app_name: str = "Asmeranda AI Backend"
            app_version: str = "0.1.0"
            debug: bool = False

            # Server
            host: str = "0.0.0.0"
            port: int = 8000

            # CORS
            cors_origins: List[str] = ["*"]

            # Storage dataset
            data_dir: Path = PROJECT_ROOT / "data"
            max_upload_size_mb: int = 200

            # Security (placeholder; diisi di F7 auth flow)
            jwt_secret: str = "change-me-in-production"
            jwt_algorithm: str = "HS256"
            jwt_expire_minutes: int = 60 * 24

            # Production safety checks
            production_mode: bool = False

            # Logging
            log_level: str = "INFO"

        return Settings()

    # Fallback: dataclass sederhana agar backend tetap jalan walau
    # pydantic-settings tidak terpasang.
    from dataclasses import dataclass

    @dataclass
    class Settings:  # type: ignore[no-redef]
        app_name: str = "Asmeranda AI Backend"
        app_version: str = "0.1.0"
        debug: bool = False
        host: str = "0.0.0.0"
        port: int = 8000
        cors_origins: List[str] = None  # type: ignore
        data_dir: Path = PROJECT_ROOT / "data"
        max_upload_size_mb: int = 200
        jwt_secret: str = "change-me-in-production"
        jwt_algorithm: str = "HS256"
        jwt_expire_minutes: int = 1440
        log_level: str = "INFO"

    s = Settings()
    s.cors_origins = ["*"]
    return s


settings = _build_settings()


# Production safety validation
def _validate_production_safety():
    """Validate critical security settings in production mode."""
    if settings.production_mode:
        # Check for insecure defaults in production
        warnings = []
        
        if settings.jwt_secret == "change-me-in-production":
            warnings.append("JWT_SECRET is using default value - set a strong secret in production")
        
        if settings.cors_origins == ["*"]:
            warnings.append("CORS_ORIGINS is set to wildcard - restrict to specific origins in production")
        
        if settings.debug:
            warnings.append("DEBUG mode is enabled in production - disable for security")
        
        if warnings:
            logger = logging.getLogger("asmeranda.backend")
            logger.warning("Production safety warnings: %s", "; ".join(warnings))


_validate_production_safety()


# Auto-create data dir
os.makedirs(settings.data_dir, exist_ok=True)
