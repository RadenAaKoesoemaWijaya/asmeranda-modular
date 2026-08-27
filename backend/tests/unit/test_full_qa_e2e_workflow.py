"""
Comprehensive QA End-to-End Pipeline Verification Test Suite.
Covers:
1. Dataset Upload & Validation
2. Data Preprocessing & Feature Selection
3. Model Training, Cross-Validation & Metric Evaluation
4. Explainable AI (SHAP & LIME)
5. Model Download (.pkl), Model Re-upload, and New Data Inference (Single & Batch File)
"""
import io
import pickle
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import dataset_service, training_service, preprocessing_service, interpretation_service
from core.state import get_state, new_state_id


@pytest.fixture
def client():
    return TestClient(app)


def test_full_e2e_qa_workflow(client):
    """Verify entire workflow from upload -> preprocessing -> feature selection -> training -> XAI -> model download & prediction."""
    # -------------------------------------------------------------
    # STAGE 1: Dataset Generation & Upload
    # -------------------------------------------------------------
    np.random.seed(42)
    n_rows = 150
    df_raw = pd.DataFrame({
        "age": np.random.randint(20, 70, size=n_rows),
        "income": np.random.uniform(20000, 120000, size=n_rows),
        "credit_score": np.random.uniform(300, 850, size=n_rows),
        "redundant_feat": np.random.randn(n_rows),
        "department": np.random.choice(["Sales", "Engineering", "Marketing", "HR"], size=n_rows),
        "churn": np.random.choice([0, 1], p=[0.7, 0.3], size=n_rows),
    })

    # Save to CSV bytes
    csv_bytes = df_raw.to_csv(index=False).encode("utf-8")
    upload_file = ("customer_data.csv", io.BytesIO(csv_bytes), "text/csv")
    
    res_upload = client.post("/api/v1/datasets", files={"file": upload_file})
    assert res_upload.status_code == 200, f"Upload failed: {res_upload.text}"
    dataset_info = res_upload.json()
    assert dataset_info["success"] is True
    metadata = dataset_info.get("metadata", {})
    dataset_id = metadata.get("dataset_id")
    assert dataset_id is not None
    assert metadata.get("rows") == n_rows

    # Verify EDA summary
    res_eda = client.get(f"/api/v1/eda/{dataset_id}/summary")
    assert res_eda.status_code == 200
    eda_data = res_eda.json()
    assert "columns" in eda_data or "numerical_columns" in eda_data or "shape" in eda_data

    # -------------------------------------------------------------
    # STAGE 2: Preprocessing & Feature Selection
    # -------------------------------------------------------------
    preproc_payload = {
        "dataset_id": dataset_id,
        "target_column": "churn",
        "problem_type": "Classification",
        "scaling_method": "robust",
        "imputation_strategy": "mean",
        "apply_encoding": True,
        "test_size": 0.2,
        "random_state": 42,
        "feature_selection": {
            "method": "kbest",
            "max_features": 4,
            "threshold": 0.05,
        },
        "imbalance_handling": {
            "method": "smote",
            "sampling_strategy": "auto",
        }
    }

    res_preproc = client.post("/api/v1/preprocessing/run", json=preproc_payload)
    assert res_preproc.status_code == 200, f"Preprocessing failed: {res_preproc.text}"
    preproc_result = res_preproc.json()
    assert preproc_result["success"] is True
    state_id = preproc_result["state_id"]
    assert state_id is not None
    assert preproc_result["n_samples_train"] > 0
    assert preproc_result["n_samples_test"] > 0

    # -------------------------------------------------------------
    # STAGE 3: Model Training & Evaluation
    # -------------------------------------------------------------
    train_payload = {
        "state_id": state_id,
        "model_type": "RandomForest",
        "problem_type": "Classification",
        "cv_method": "stratified",
        "cv_folds": 5,
        "hyperparams": {
            "n_estimators": 50,
            "max_depth": 5,
        },
    }

    res_train = client.post("/api/v1/training/start", json=train_payload)
    assert res_train.status_code == 200, f"Training failed: {res_train.text}"
    train_result = res_train.json()
    assert train_result["success"] is True
    model_id = train_result["model_id"]
    assert model_id is not None
    assert "metrics" in train_result
    assert "accuracy" in train_result["metrics"]
    assert "f1_macro" in train_result["metrics"]
    assert "cv_scores" in train_result

    # -------------------------------------------------------------
    # STAGE 4: Explainable AI (SHAP & LIME)
    # -------------------------------------------------------------
    # SHAP
    shap_payload = {
        "model_id": model_id,
        "state_id": state_id,
        "max_samples": 50,
    }
    res_shap = client.post("/api/v1/interpretation/shap", json=shap_payload)
    assert res_shap.status_code == 200
    shap_res = res_shap.json()
    assert shap_res["success"] is True
    assert "feature_importance" in shap_res

    # LIME
    lime_payload = {
        "model_id": model_id,
        "state_id": state_id,
        "sample_index": 0,
        "num_features": 4,
    }
    res_lime = client.post("/api/v1/interpretation/lime", json=lime_payload)
    assert res_lime.status_code == 200
    lime_res = res_lime.json()
    assert lime_res["success"] is True

    # -------------------------------------------------------------
    # STAGE 5: Model Download (.pkl) & Re-uploading Model
    # -------------------------------------------------------------
    res_download = client.get(f"/api/v1/training/models/{model_id}/download")
    assert res_download.status_code == 200
    downloaded_model_bytes = res_download.content
    assert len(downloaded_model_bytes) > 0

    # Test Re-upload of downloaded model
    upload_model_file = ("downloaded_model.pkl", io.BytesIO(downloaded_model_bytes), "application/octet-stream")
    res_reupload = client.post("/api/v1/training/models/upload", files={"file": upload_model_file})
    assert res_reupload.status_code == 200, f"Model upload failed: {res_reupload.text}"
    reupload_result = res_reupload.json()
    assert reupload_result["success"] is True
    uploaded_model_id = reupload_result["model_id"]
    assert uploaded_model_id is not None

    # -------------------------------------------------------------
    # STAGE 6: Inference on New Data (Single Record & Batch File)
    # -------------------------------------------------------------
    # 6A: Single Record Prediction
    feature_names = train_result.get("feature_importances", [])
    single_record = {f["feature"]: 1.0 for f in feature_names} if feature_names else {"age": 35, "income": 50000}
    res_pred_single = client.post(
        f"/api/v1/training/models/{uploaded_model_id}/predict",
        json={"data": [single_record]},
    )
    assert res_pred_single.status_code == 200
    pred_single_data = res_pred_single.json()
    assert pred_single_data["success"] is True
    assert len(pred_single_data["predictions"]) == 1
    assert pred_single_data["probabilities"] is not None

    # 6B: Batch File Prediction (CSV upload with new data)
    new_test_df = pd.DataFrame({
        "age": [25, 45, 60, 32],
        "income": [30000, 85000, 110000, 48000],
        "credit_score": [650, 780, 820, 590],
        "department": ["Sales", "Engineering", "Marketing", "HR"],
    })
    batch_csv_bytes = new_test_df.to_csv(index=False).encode("utf-8")
    batch_file_payload = ("new_unseen_data.csv", io.BytesIO(batch_csv_bytes), "text/csv")

    res_pred_batch = client.post(
        f"/api/v1/training/models/{uploaded_model_id}/predict-file",
        files={"file": batch_file_payload},
    )
    assert res_pred_batch.status_code == 200, f"Batch predict failed: {res_pred_batch.text}"
    pred_batch_data = res_pred_batch.json()
    assert pred_batch_data["success"] is True
    assert pred_batch_data["total_rows"] == 4
    assert len(pred_batch_data["predictions"]) == 4
    assert "preview" in pred_batch_data
    assert "Prediction" in pred_batch_data["preview"][0]
