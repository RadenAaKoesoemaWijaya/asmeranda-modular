"""
Workflow Validator module for Asmeranda AI.

Validates prerequisites and state transitions between workflow steps:
1. upload_to_eda
2. eda_to_preprocessing
3. preprocessing_to_training
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class WorkflowValidator:
    """Validator for ML workflow step transitions."""

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        self.state: Dict[str, Any] = state.copy() if isinstance(state, dict) else {}
        self.validation_rules: Dict[str, List[str]] = {
            "upload_to_eda": ["dataset_id"],
            "eda_to_preprocessing": ["dataset_id"],
            "preprocessing_to_optimization": [
                "dataset_id",
                "state_id",
                "problem_type",
            ],
            "preprocessing_to_training": [
                "dataset_id",
                "state_id",
                "problem_type",
            ],
            "training_to_evaluation": ["model_id", "model_type"],
            "training_to_xai": ["model_id", "model_type"],
            "preprocessing_to_clustering": ["dataset_id", "state_id"],
            "eda_to_clustering": ["dataset_id"],
        }

    def validate(self, step_name: str) -> Dict[str, Any]:
        """
        Validate if the state meets requirements for the given workflow step.
        
        Parameters
        ----------
        step_name : str
            Step transition name
            
        Returns
        -------
        dict
            Validation result containing 'valid', 'errors', 'message', 'error_id', and 'step'
        """
        required_fields = self.validation_rules.get(step_name, [])
        errors: List[str] = []
        problem_type = self.state.get("problem_type")

        if step_name == "upload_to_eda":
            if not self.state.get("dataset_id"):
                errors.append("Dataset belum diunggah atau dataset_id tidak valid.")

        elif step_name == "eda_to_preprocessing":
            if not self.state.get("dataset_id"):
                errors.append("Dataset belum dipilih.")

        elif step_name == "preprocessing_to_optimization":
            if not self.state.get("dataset_id"):
                errors.append("Dataset belum dipilih.")
            if not self.state.get("state_id"):
                errors.append("Tahap preprocessing belum selesai.")
            if problem_type in ("Clustering", "Unsupervised"):
                errors.append("Optimasi hyperparameter hanya tersedia untuk Supervised Learning (Klasifikasi / Regresi).")
            elif not self.state.get("target_column") and problem_type in ("Classification", "Regression"):
                errors.append("Kolom target belum ditentukan.")

        elif step_name == "preprocessing_to_training":
            if not self.state.get("dataset_id"):
                errors.append("Dataset belum dipilih.")
            if not self.state.get("state_id"):
                errors.append("Tahap preprocessing belum selesai.")
            if problem_type in ("Clustering", "Unsupervised"):
                errors.append("Pelatihan model terarah hanya untuk Supervised Learning. Gunakan menu Clustering untuk Unsupervised.")
            elif not self.state.get("target_column") and problem_type in ("Classification", "Regression"):
                errors.append("Kolom target belum ditentukan.")
            if self.state.get("n_samples_train") is None:
                errors.append("Data training belum siap.")

        elif step_name in ("training_to_evaluation", "training_to_xai"):
            if not self.state.get("model_id"):
                errors.append("Model belum dilatih atau model_id tidak ditemukan.")

        elif step_name in ("preprocessing_to_clustering", "eda_to_clustering"):
            if not self.state.get("dataset_id"):
                errors.append("Dataset belum dipilih untuk analisis clustering.")
            if step_name == "preprocessing_to_clustering" and not self.state.get("state_id"):
                errors.append("Preprocessing data belum selesai.")

        else:
            # Generic field checking for custom steps
            for field in required_fields:
                if self.state.get(field) is None:
                    errors.append(f"Field '{field}' wajib diisi sebelum melanjutkan.")

        is_valid = len(errors) == 0
        return {
            "valid": is_valid,
            "step": step_name,
            "errors": errors,
            "error_id": f"ERR_WORKFLOW_{step_name.upper()}" if not is_valid else None,
            "message": "Validasi berhasil" if is_valid else "; ".join(errors),
        }
