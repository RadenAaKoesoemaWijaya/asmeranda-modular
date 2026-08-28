"""
Test suite for Production Readiness Security and Operational Hardening.

Tests:
1. Cryptographic HMAC-SHA256 Model Signing & Tamper Detection
2. Persistent Database-Backed User Authentication & Role Verification
3. Secure JSON Session Storage
4. State Parquet Partition Persistence & Recovery
"""
import json
import pickle
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from backend.core.model_security import (
    sign_model_file,
    verify_model_integrity,
    compute_file_hmac,
)
from backend.core.auth import (
    UserCreate,
    UserRole,
    user_store,
    verify_password,
    get_password_hash,
)
from backend.core.session_manager import SessionManager
from backend.core.state import (
    get_state,
    set_state,
    _save_state_metadata_to_disk,
    _load_state_from_disk,
    _PARTITIONS_DIR,
)
from backend.core.config import settings


class TestModelCryptographicSecurity:
    """Test HMAC-SHA256 integrity verification for model serialization."""

    def test_sign_and_verify_valid_model(self, tmp_path):
        model_file = tmp_path / "model_test_01.pkl"
        dummy_model = {"model_type": "RandomForest", "weights": [1, 2, 3, 4]}
        with open(model_file, "wb") as f:
            pickle.dump(dummy_model, f)

        # Sign the model
        sig_file = sign_model_file(model_file, metadata={"model_id": "test_01"})
        assert sig_file.exists()
        assert sig_file.suffix == ".sig"

        # Verify integrity
        is_valid, message = verify_model_integrity(model_file)
        assert is_valid is True
        assert "verified successfully" in message

    def test_tampered_model_rejected(self, tmp_path):
        model_file = tmp_path / "model_tampered.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"model": "legitimate_model"}, f)

        # Sign legitimate model
        sign_model_file(model_file)

        # Attacker tampers with the model binary payload
        with open(model_file, "ab") as f:
            f.write(b"\x00\xffMALICIOUS_OPCODE_INJECTION")

        # Verify integrity should detect tampering
        is_valid, message = verify_model_integrity(model_file)
        assert is_valid is False
        assert "Tampered model detected" in message

    def test_unsigned_model_in_production(self, tmp_path, monkeypatch):
        model_file = tmp_path / "unsigned_model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump({"model": "raw"}, f)

        # In production mode, unsigned models must be rejected
        monkeypatch.setattr(settings, "production_mode", True)
        is_valid, message = verify_model_integrity(model_file, allow_legacy_unsigned=False)
        assert is_valid is False
        assert "Missing cryptographic signature" in message


class TestPersistentDatabaseAuthentication:
    """Test DB-backed user storage and security policy."""

    def test_admin_initialization(self):
        admin_user = user_store.get_user_by_username("admin")
        assert admin_user is not None
        assert admin_user.role == UserRole.ADMIN
        assert admin_user.is_active is True

    def test_create_and_authenticate_user(self):
        test_username = "lead_analyst_2026"
        test_pwd = "Secure@AnalystPass2026!"
        
        # Cleanup if exists
        user_in = UserCreate(
            username=test_username,
            password=test_pwd,
            email="analyst@asmeranda.ai",
            role=UserRole.ANALYST,
        )
        
        user_created = user_store.create_user(user_in)
        assert user_created.username == test_username
        assert user_created.role == UserRole.ANALYST
        
        # Verify persistence and retrieval
        retrieved = user_store.get_user_by_username(test_username)
        assert retrieved is not None
        assert verify_password(test_pwd, retrieved.hashed_password) is True
        assert verify_password("WrongPassword!", retrieved.hashed_password) is False

    def test_duplicate_user_rejected(self):
        user_in = UserCreate(
            username="admin",
            password="SomePassword123!",
            role=UserRole.ANALYST,
        )
        with pytest.raises(ValueError, match="sudah digunakan"):
            user_store.create_user(user_in)


class TestSecureSessionManager:
    """Test secure JSON session management."""

    def test_session_creation_and_json_persistence(self, tmp_path):
        mgr = SessionManager(session_timeout_minutes=15, storage_dir=tmp_path)
        sid = mgr.create_session(user_id="usr-test-100", metadata={"role": "analyst"})

        assert mgr.validate_session(sid) is True
        info = mgr.get_session_info(sid)
        assert info["user_id"] == "usr-test-100"

        # Check that sessions.json was written (not pickle)
        json_path = tmp_path / "sessions.json"
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert sid in data


class TestStateParquetPartitioning:
    """Test DataFrame/Series state persistence across container restarts."""

    def test_state_parquet_save_and_reload(self, tmp_path):
        state_id = "state_resilience_test_01"
        
        df_train = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [10.0, 20.0, 30.0]})
        s_target = pd.Series([0, 1, 0], name="target")

        # Set state
        set_state(
            state_id,
            problem_type="Classification",
            target_column="target",
            X_train=df_train,
            y_train=s_target,
        )

        # Simulate fresh reload by creating a clean dictionary and loading from disk
        loaded_state = _load_state_from_disk(state_id)
        assert loaded_state is not None
        assert loaded_state.get("problem_type") == "Classification"
        
        # Verify heavy DataFrame was restored from Parquet
        reloaded_X = loaded_state.get("X_train")
        assert isinstance(reloaded_X, pd.DataFrame)
        assert len(reloaded_X) == 3
        assert list(reloaded_X.columns) == ["feat1", "feat2"]

        reloaded_y = loaded_state.get("y_train")
        assert reloaded_y is not None
        assert len(reloaded_y) == 3
