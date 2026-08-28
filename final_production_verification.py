"""
Comprehensive Production Readiness & Security Verification Script
Using standard unittest runner for deterministic execution.
"""
import json
import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Ensure root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
)
from backend.services import training_service
from backend.core.config import settings


class ProductionSecurityVerificationTests(unittest.TestCase):
    """Production readiness verification test suite."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_model_hmac_signature_and_tamper_detection(self):
        """Verify HMAC-SHA256 model signing and detection of tampered binary payload."""
        model_file = self.temp_path / "model_rf.pkl"
        payload = {"model_type": "RandomForest", "weights": [0.1, 0.4, 0.5]}
        with open(model_file, "wb") as f:
            pickle.dump(payload, f)

        # 1. Sign model
        sig_file = sign_model_file(model_file, metadata={"model_id": "rf_01"})
        self.assertTrue(sig_file.exists())
        self.assertEqual(sig_file.suffix, ".sig")

        # 2. Verify valid model
        is_valid, msg = verify_model_integrity(model_file)
        self.assertTrue(is_valid, f"Verification failed for legitimate model: {msg}")

        # 3. Simulate attacker modifying the .pkl bytes
        with open(model_file, "ab") as f:
            f.write(b"\x00\xfeTAMPERED_PAYLOAD")

        is_valid_tampered, tamper_msg = verify_model_integrity(model_file)
        self.assertFalse(is_valid_tampered, "Tampered model was not rejected!")
        self.assertIn("Tampered model detected", tamper_msg)

    def test_02_persistent_database_user_store(self):
        """Verify user creation, password bcrypt hashing, persistence, and uniqueness."""
        # 1. Verify default admin initialized
        admin = user_store.get_user_by_username("admin")
        self.assertIsNotNone(admin)
        self.assertEqual(admin.role, UserRole.ADMIN)

        # 2. Create distinct analyst user
        uname = f"qa_lead_{os.urandom(4).hex()}"
        pwd = "Secure@QAPassword2026!"
        created = user_store.create_user(
            UserCreate(
                username=uname,
                password=pwd,
                email=f"{uname}@asmeranda.ai",
                role=UserRole.ANALYST,
            )
        )
        self.assertEqual(created.username, uname)
        self.assertEqual(created.role, UserRole.ANALYST)

        # 3. Fetch from DB
        fetched = user_store.get_user_by_username(uname)
        self.assertIsNotNone(fetched)
        self.assertTrue(verify_password(pwd, fetched.hashed_password))
        self.assertFalse(verify_password("WrongPassword!", fetched.hashed_password))

        # 4. Duplicate rejection
        with self.assertRaises(ValueError):
            user_store.create_user(
                UserCreate(username=uname, password="AnotherPassword123!", role=UserRole.VIEWER)
            )

    def test_03_secure_json_session_storage(self):
        """Verify session manager uses JSON storage with expiration and datetime parsing."""
        session_mgr = SessionManager(session_timeout_minutes=30, storage_dir=self.temp_path)
        sid = session_mgr.create_session(user_id="usr-qa-test", metadata={"env": "prod"})
        
        # Verify JSON file written
        json_file = self.temp_path / "sessions.json"
        self.assertTrue(json_file.exists())

        # Validate session
        self.assertTrue(session_mgr.validate_session(sid))
        info = session_mgr.get_session_info(sid)
        self.assertEqual(info["user_id"], "usr-qa-test")

    def test_04_state_parquet_persistence_resilience(self):
        """Verify DataFrame and Series tabular partitions survive restart via Parquet."""
        state_id = f"test_state_{os.urandom(4).hex()}"
        df_train = pd.DataFrame({"x1": [10.5, 20.2, 30.1], "x2": [100.0, 200.0, 300.0]})
        s_target = pd.Series([1, 0, 1], name="label")

        set_state(
            state_id,
            problem_type="Classification",
            target_column="label",
            X_train=df_train,
            y_train=s_target,
        )

        # Simulate cold restart load from disk
        restored_state = _load_state_from_disk(state_id)
        self.assertIsNotNone(restored_state)
        self.assertEqual(restored_state.get("problem_type"), "Classification")

        # Verify heavy dataframe restored
        reloaded_X = restored_state.get("X_train")
        self.assertIsInstance(reloaded_X, pd.DataFrame)
        self.assertEqual(len(reloaded_X), 3)
        self.assertListEqual(list(reloaded_X.columns), ["x1", "x2"])

        reloaded_y = restored_state.get("y_train")
        self.assertIsNotNone(reloaded_y)
        self.assertEqual(len(reloaded_y), 3)

    def test_05_training_service_model_signing_lifecycle(self):
        """Verify training_service trains model, signs .pkl, and verifies on load."""
        df_X = pd.DataFrame({"feature1": np.random.randn(50), "feature2": np.random.randn(50)})
        s_y = pd.Series(np.random.randint(0, 2, 50))

        res = training_service.train(
            X_train=df_X,
            y_train=s_y,
            X_test=df_X.head(10),
            y_test=s_y.head(10),
            model_type="LogisticRegression",
            problem_type="Classification",
        )
        self.assertTrue(res.get("success"), f"Training failed: {res.get('error')}")
        model_id = res.get("model_id")

        # Verify signature file created
        model_path = Path(settings.data_dir) / "models" / f"{model_id}.pkl"
        sig_path = Path(settings.data_dir) / "models" / f"{model_id}.pkl.sig"
        self.assertTrue(model_path.exists())
        self.assertTrue(sig_path.exists())

        # Load model with signature verification
        loaded = training_service.load_model(model_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.get("model_type"), "LogisticRegression")

        # Cleanup
        training_service.delete_model(model_id)
        self.assertFalse(model_path.exists())
        self.assertFalse(sig_path.exists())


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(ProductionSecurityVerificationTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
