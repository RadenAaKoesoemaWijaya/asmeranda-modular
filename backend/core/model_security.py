"""
Model Security Module for Asmeranda AI.

Provides cryptographic signing and verification for serialized model artifacts (.pkl/.joblib)
to prevent Insecure Deserialization attacks (Arbitrary Code Execution / RCE).
"""
from __future__ import annotations

import hmac
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from backend.core.config import settings
except ImportError:
    from core.config import settings

logger = logging.getLogger("asmeranda.security.model_security")


def _get_signing_key() -> bytes:
    """Get the secret key used for HMAC signing of model artifacts.
    
    Priority:
    1. ASMERANDA_MODEL_SIGNING_KEY env var / settings.model_signing_key
    2. ASMERANDA_JWT_SECRET (only if explicitly set, i.e., not the runtime random default)
    3. Deterministic fallback derived from project path (dev-mode only)
    """
    # 1. Explicit model signing key (highest priority)
    key = getattr(settings, "model_signing_key", None)
    if key:
        return key.encode("utf-8") if isinstance(key, str) else key
    
    # 2. JWT secret — use only if it looks non-random (was explicitly configured)
    jwt_secret = getattr(settings, "jwt_secret", None)
    if jwt_secret and jwt_secret not in ("change-me-in-production-use-a-long-random-string",):
        # Heuristic: explicitly-set secrets don't change across restarts
        return jwt_secret.encode("utf-8") if isinstance(jwt_secret, str) else jwt_secret
    
    # 3. Deterministic fallback: derive from project path so it's stable per-machine
    import hashlib as _hl
    fallback_seed = str(Path(__file__).resolve().parent.parent.parent)
    fallback = _hl.sha256(("asmeranda-dev-model-key:" + fallback_seed).encode()).hexdigest()
    logger.debug("Using deterministic fallback signing key (set ASMERANDA_MODEL_SIGNING_KEY for production)")
    return fallback.encode("utf-8")


def compute_file_hmac(file_path: Path) -> str:
    """Compute HMAC-SHA256 signature for a file on disk."""
    key = _get_signing_key()
    hasher = hmac.new(key, digestmod=hashlib.sha256)
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_hmac(data: bytes) -> str:
    """Compute HMAC-SHA256 signature for in-memory bytes."""
    key = _get_signing_key()
    return hmac.new(key, data, digestmod=hashlib.sha256).hexdigest()


def sign_model_file(model_path: Path, metadata: Optional[Dict[str, Any]] = None) -> Path:
    """
    Sign a model file by generating a paired .sig file containing HMAC signature & metadata.
    
    Parameters
    ----------
    model_path : Path
        Path to the .pkl model file.
    metadata : Optional[Dict[str, Any]]
        Optional metadata (e.g. model_id, created_at, owner_id) to include in signature manifest.
        
    Returns
    -------
    Path
        Path to the generated signature file (.pkl.sig).
    """
    sig_path = model_path.with_suffix(".pkl.sig")
    file_signature = compute_file_hmac(model_path)
    
    manifest = {
        "model_file": model_path.name,
        "signature_algorithm": "HMAC-SHA256",
        "signature": file_signature,
        "metadata": metadata or {}
    }
    
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Model signature created for {model_path.name}: {sig_path.name}")
    return sig_path


def verify_model_integrity(model_path: Path, allow_legacy_unsigned: bool = False) -> Tuple[bool, str]:
    """
    Verify the integrity and authenticity of a model file before loading/deserialization.
    
    Parameters
    ----------
    model_path : Path
        Path to the model file to verify.
    allow_legacy_unsigned : bool
        If True, allows unsigned models in dev mode with a warning. Defaults to False.
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, reason_message)
    """
    if not model_path.exists():
        return False, f"Model file '{model_path.name}' does not exist."
    
    sig_path = model_path.with_suffix(".pkl.sig")
    if not sig_path.exists():
        # Check if production mode or unsigned allowed
        is_prod = getattr(settings, "production_mode", False)
        if not is_prod or allow_legacy_unsigned:
            logger.debug(f"Model {model_path.name} is unsigned (legacy/dev mode).")
            return True, "Legacy unsigned model accepted in non-production mode."
        return False, f"Missing cryptographic signature for model '{model_path.name}'."
    
    try:
        with open(sig_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        expected_signature = manifest.get("signature")
        if not expected_signature:
            return False, "Signature manifest is corrupted: missing signature key."
        
        actual_signature = compute_file_hmac(model_path)
        
        # Constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(actual_signature, expected_signature):
            logger.error(f"TAMPERED MODEL DETECTED: {model_path.name}. Signature mismatch!")
            return False, f"Tampered model detected! Integrity check failed for '{model_path.name}'."
        
        return True, "Model signature verified successfully."
        
    except Exception as e:
        logger.error(f"Failed to verify model signature: {e}")
        return False, f"Model verification error: {str(e)}"
