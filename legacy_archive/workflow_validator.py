"""
Workflow validator untuk Asmeranda.

Semua method menerima ``state`` (dict) secara eksplisit. Untuk
backward compatibility dengan legacy UI, ``WorkflowValidator()``
otomatis memakai ``st.session_state`` ketika state tidak diberikan.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from core.state import get_state

try:
    import streamlit as _st  # type: ignore

    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    _st = None
    _ST_AVAILABLE = False


# Tipe untuk check function
CheckFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def _has_value(state: Dict[str, Any], key: str) -> bool:
    """Check if a session state key exists and is not None/empty."""
    if key not in state:
        return False
    value = state[key]
    if value is None:
        return False
    if isinstance(value, (list, dict, str)) and len(value) == 0:
        return False
    return True


class WorkflowValidator:
    """Comprehensive workflow validation with detailed error reporting."""

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        if state is None:
            state = get_state()
        self.state = state
        self.validation_rules: Dict[str, Dict[str, Any]] = {
            "upload_to_eda": {
                "required": ["data"],
                "checks": [
                    self._check_data_not_empty,
                    self._check_minimum_rows,
                    self._check_minimum_columns,
                ],
            },
            "eda_to_preprocessing": {
                "required": ["data", "numerical_columns", "categorical_columns"],
                "checks": [
                    self._check_column_consistency,
                    self._check_target_column_validity,
                    self._check_data_quality,
                ],
            },
            "preprocessing_to_training": {
                "required": ["X_train", "X_test", "y_train", "y_test", "problem_type"],
                "checks": [
                    self._check_train_test_split,
                    self._check_feature_target_consistency,
                    self._check_problem_type_validity,
                ],
            },
            "training_to_interpretation": {
                "required": ["model_results"],
                "checks": [
                    self._check_model_results_validity,
                    self._check_model_availability_for_interpretation,
                ],
            },
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_state(self, state: Dict[str, Any]) -> None:
        """Ganti state yang dirujuk validator (untuk multi-user)."""
        self.state = state

    def validate_workflow_transition(self, from_step: str, to_step: str) -> Dict[str, Any]:
        """Validate workflow transition between steps."""
        transition_key = f"{from_step}_to_{to_step}"

        if transition_key not in self.validation_rules:
            return {
                "valid": False,
                "errors": [f"Unknown workflow transition: {transition_key}"],
                "warnings": [],
                "recommendations": [],
                "missing": {},
            }

        rule = self.validation_rules[transition_key]
        result: Dict[str, Any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "missing": {},
        }

        for required_key in rule["required"]:
            if not _has_value(self.state, required_key):
                msg = f"Missing required data: {required_key}"
                result["errors"].append(msg)
                result["missing"][required_key] = msg
                result["valid"] = False

        for check_func in rule["checks"]:
            try:
                check_result = check_func()
                if not check_result.get("valid", False):
                    result["valid"] = False
                    result["errors"].extend(check_result.get("errors", []))
                result["warnings"].extend(check_result.get("warnings", []))
                result["recommendations"].extend(check_result.get("recommendations", []))
            except Exception as exc:
                result["valid"] = False
                result["errors"].append(f"Validation check failed: {exc}")

        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------
    def _check_data_not_empty(self) -> Dict[str, Any]:
        data = self.state.get("data")
        if data is None or (hasattr(data, "__len__") and len(data) == 0):
            return {
                "valid": False,
                "errors": ["Dataset is empty or not loaded"],
                "warnings": [],
                "recommendations": ["Please upload a valid dataset"],
            }
        return {"valid": True, "errors": [], "warnings": [], "recommendations": []}

    def _check_minimum_rows(self) -> Dict[str, Any]:
        data = self.state.get("data")
        if data is not None and hasattr(data, "__len__") and len(data) < 10:
            return {
                "valid": False,
                "errors": ["Dataset has insufficient rows (minimum 10 required)"],
                "warnings": [],
                "recommendations": ["Upload a dataset with at least 10 rows"],
            }
        return {"valid": True, "errors": [], "warnings": [], "recommendations": []}

    def _check_minimum_columns(self) -> Dict[str, Any]:
        data = self.state.get("data")
        if data is not None and hasattr(data, "columns") and len(data.columns) < 2:
            return {
                "valid": False,
                "errors": ["Dataset needs at least 2 columns (1 feature + 1 target)"],
                "warnings": [],
                "recommendations": ["Upload a dataset with multiple columns"],
            }
        return {"valid": True, "errors": [], "warnings": [], "recommendations": []}

    def _check_column_consistency(self) -> Dict[str, Any]:
        data = self.state.get("data")
        numerical_cols = self.state.get("numerical_columns", []) or []
        categorical_cols = self.state.get("categorical_columns", []) or []

        errors: List[str] = []
        warnings: List[str] = []

        if data is not None and hasattr(data, "columns"):
            missing_numerical = [c for c in numerical_cols if c not in data.columns]
            missing_categorical = [c for c in categorical_cols if c not in data.columns]
            if missing_numerical:
                errors.append(f"Numerical columns not found in data: {missing_numerical}")
            if missing_categorical:
                errors.append(f"Categorical columns not found in data: {missing_categorical}")

            overlap = set(numerical_cols) & set(categorical_cols)
            if overlap:
                warnings.append(f"Columns appear in both numerical and categorical lists: {overlap}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "recommendations": [],
        }

    def _check_target_column_validity(self) -> Dict[str, Any]:
        data = self.state.get("data")
        target_column = self.state.get("target_column")

        if data is not None and target_column:
            if target_column not in data.columns:
                return {
                    "valid": False,
                    "errors": [f"Target column '{target_column}' not found in dataset"],
                    "warnings": [],
                    "recommendations": ["Select a valid target column"],
                }

            target_data = data[target_column]
            try:
                null_percentage = float(target_data.isnull().sum() / len(target_data))
            except Exception:
                null_percentage = 0.0

            if null_percentage > 0.5:
                return {
                    "valid": False,
                    "errors": [f"Target column has too many missing values ({null_percentage:.1%})"],
                    "warnings": [],
                    "recommendations": ["Handle missing values in target column or choose different target"],
                }

        return {"valid": True, "errors": [], "warnings": [], "recommendations": []}

    def _check_data_quality(self) -> Dict[str, Any]:
        data = self.state.get("data")
        if data is None:
            return {"valid": True, "errors": [], "warnings": [], "recommendations": []}

        warnings: List[str] = []
        recommendations: List[str] = []

        try:
            missing_pct = float(data.isnull().sum().sum() / (len(data) * len(data.columns)))
        except Exception:
            missing_pct = 0.0
        if missing_pct > 0.3:
            warnings.append(f"High percentage of missing values ({missing_pct:.1%})")
            recommendations.append("Consider imputation or removal of columns with many missing values")

        try:
            duplicate_pct = float(data.duplicated().sum() / len(data))
        except Exception:
            duplicate_pct = 0.0
        if duplicate_pct > 0.1:
            warnings.append(f"High percentage of duplicate rows ({duplicate_pct:.1%})")
            recommendations.append("Consider removing duplicate rows")

        return {
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "recommendations": recommendations,
        }

    def _check_train_test_split(self) -> Dict[str, Any]:
        X_train = self.state.get("X_train")
        X_test = self.state.get("X_test")
        y_train = self.state.get("y_train")
        y_test = self.state.get("y_test")

        errors: List[str] = []
        components = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
        for name, comp in components.items():
            if comp is None:
                errors.append(f"Missing {name}")
            elif hasattr(comp, "__len__") and len(comp) == 0:
                errors.append(f"Empty {name}")

        if X_train is not None and y_train is not None:
            try:
                if len(X_train) != len(y_train):
                    errors.append("X_train and y_train have different lengths")
            except Exception:
                pass

        if X_test is not None and y_test is not None:
            try:
                if len(X_test) != len(y_test):
                    errors.append("X_test and y_test have different lengths")
            except Exception:
                pass

        return {"valid": len(errors) == 0, "errors": errors, "warnings": [], "recommendations": []}

    def _check_feature_target_consistency(self) -> Dict[str, Any]:
        X_train = self.state.get("X_train")
        X_test = self.state.get("X_test")
        errors: List[str] = []
        if X_train is not None and X_test is not None:
            if hasattr(X_train, "columns") and hasattr(X_test, "columns"):
                if list(X_train.columns) != list(X_test.columns):
                    errors.append("X_train and X_test have different feature columns")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": [], "recommendations": []}

    def _check_problem_type_validity(self) -> Dict[str, Any]:
        problem_type = self.state.get("problem_type")
        y_train = self.state.get("y_train")

        if problem_type is None:
            return {
                "valid": False,
                "errors": ["Problem type not specified"],
                "warnings": [],
                "recommendations": ["Select a problem type (Classification/Regression/Forecasting)"],
            }

        if y_train is not None and problem_type in ("Classification", "Regression"):
            try:
                unique_values = int(pd.Series(y_train).nunique())
            except Exception:
                unique_values = 0

            if problem_type == "Classification" and unique_values < 2:
                return {
                    "valid": False,
                    "errors": ["Classification requires at least 2 unique target values"],
                    "warnings": [],
                    "recommendations": ["Check target column or change problem type"],
                }
            if problem_type == "Regression" and unique_values < 10:
                return {
                    "valid": False,
                    "errors": ["Regression requires continuous target values"],
                    "warnings": [],
                    "recommendations": ["Ensure target column is continuous or change problem type"],
                }

        return {"valid": True, "errors": [], "warnings": [], "recommendations": []}

    def _check_model_results_validity(self) -> Dict[str, Any]:
        model_results = self.state.get("model_results")
        if not model_results:
            return {
                "valid": False,
                "errors": ["No trained models found"],
                "warnings": [],
                "recommendations": ["Train at least one model before proceeding to interpretation"],
            }
        return {"valid": True, "errors": [], "warnings": [], "recommendations": []}

    def _check_model_availability_for_interpretation(self) -> Dict[str, Any]:
        model_results = self.state.get("model_results")
        if not model_results:
            return {"valid": True, "errors": [], "warnings": [], "recommendations": []}

        warnings: List[str] = []
        interpretable_models = [
            "Random Forest",
            "Gradient Boosting",
            "Logistic Regression",
            "Linear Regression",
        ]
        available = any(r.get("model_type") in interpretable_models for r in model_results)
        if not available:
            warnings.append(
                "No interpretable models found. Consider training Random Forest, "
                "Gradient Boosting, or Linear models"
            )
        return {
            "valid": True,
            "errors": [],
            "warnings": warnings,
            "recommendations": ["Train interpretable models for better SHAP/LIME analysis"],
        }

    # ------------------------------------------------------------------
    # Pure-data validations (dipakai di endpoint FastAPI)
    # ------------------------------------------------------------------
    def validate_data_readiness(
        self,
        data: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if len(data) >= 100:
            results.append(
                {"status": "success", "message": f"Dataset has sufficient rows ({len(data)}) for training."}
            )
        elif len(data) >= 20:
            results.append(
                {
                    "status": "warning",
                    "message": f"Dataset is relatively small ({len(data)} rows). Consider cross-validation.",
                }
            )
        else:
            results.append(
                {
                    "status": "error",
                    "message": f"Dataset is very small ({len(data)} rows). Machine learning might not be effective.",
                }
            )

        if len(data.columns) >= 3:
            results.append(
                {
                    "status": "success",
                    "message": f"Dataset has {len(data.columns)} columns, sufficient for feature engineering.",
                }
            )
        else:
            results.append(
                {
                    "status": "warning",
                    "message": "Dataset has very few columns. Limit feature engineering potential.",
                }
            )

        missing_pct = float(data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        if missing_pct == 0:
            results.append({"status": "success", "message": "No missing values detected in the dataset."})
        elif missing_pct < 0.1:
            results.append(
                {"status": "success", "message": f"Low percentage of missing values ({missing_pct:.1%}). Easy to handle."}
            )
        elif missing_pct < 0.3:
            results.append(
                {"status": "warning", "message": f"Moderate percentage of missing values ({missing_pct:.1%}). Imputation recommended."}
            )
        else:
            results.append(
                {"status": "error", "message": f"High percentage of missing values ({missing_pct:.1%}). Data cleaning is critical."}
            )

        if len(numerical_cols) > 0:
            results.append(
                {"status": "success", "message": "Numeric columns detected, suitable for regression or time series."}
            )
        if len(categorical_cols) > 0:
            results.append(
                {"status": "success", "message": "Categorical columns detected, suitable for classification."}
            )

        return results

    def check_ml_readiness(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        errors = [r for r in validation_results if r.get("status") == "error"]
        warnings = [r for r in validation_results if r.get("status") == "warning"]

        available_workflows = ["Exploratory Data Analysis"]
        if not errors:
            available_workflows.extend(["Preprocessing", "Model Training", "Model Interpretation"])

        ready = len(errors) == 0
        if ready and not warnings:
            message = "Dataset is highly optimal for machine learning."
        elif ready:
            message = "Dataset is ready for machine learning with some considerations."
        else:
            message = "Dataset requires significant cleaning before machine learning."

        recommendations: List[str] = []
        for r in errors + warnings:
            msg = r.get("message", "").lower()
            if "imputation" in msg or "missing" in msg:
                recommendations.append("Apply missing value imputation in Preprocessing tab.")
            if "small" in msg:
                recommendations.append("Use robust models like Random Forest or Cross-Validation.")
            if "cleaning" in msg:
                recommendations.append("Review data quality in EDA tab.")

        return {
            "ready": ready,
            "message": message,
            "available_workflows": available_workflows,
            "recommendations": recommendations,
        }

    def validate_eda_completeness(
        self,
        data: pd.DataFrame,
        numerical_cols: List[str],
        categorical_cols: List[str],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        missing_count = int(data.isnull().sum().sum())
        if missing_count == 0:
            results.append({"status": "success", "message": "Dataset is clean (no missing values)."})
        else:
            results.append(
                {
                    "status": "warning",
                    "message": f"Dataset has {missing_count} missing values. Preprocessing recommended.",
                }
            )

        if len(numerical_cols) > 1:
            results.append(
                {
                    "status": "success",
                    "message": f"Correlation analysis possible for {len(numerical_cols)} numerical features.",
                }
            )
        else:
            results.append(
                {"status": "warning", "message": "Too few numerical features for correlation analysis."}
            )

        if len(numerical_cols) > 0:
            results.append(
                {"status": "success", "message": "Distribution analysis available for numerical features."}
            )
            results.append(
                {
                    "status": "info",
                    "message": "Consider checking for outliers in numerical features before training.",
                }
            )

        return results

    def check_eda_readiness(self, eda_validation: List[Dict[str, Any]]) -> Dict[str, Any]:
        warnings = [r for r in eda_validation if r.get("status") == "warning"]
        recommendations: List[str] = []
        for r in warnings:
            msg = r.get("message", "").lower()
            if "missing" in msg:
                recommendations.append("Handle missing values in the Preprocessing tab.")
            if "correlation" in msg:
                recommendations.append("Add more numerical features if possible for better insight.")

        return {
            "ready": True,
            "message": "Exploratory Data Analysis is sufficient for basic ML transition.",
            "available_transitions": ["Preprocessing", "Feature Engineering"],
            "recommendations": recommendations,
        }

    def validate_ml_training_readiness(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: str,
        model_type: str,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if X_train is None or y_train is None:
            results.append({"status": "error", "message": "Training data (X or y) is missing."})
            return results

        if len(X_train) < 10:
            results.append({"status": "error", "message": f"Insufficient data for training ({len(X_train)} samples)."})
        elif len(X_train) < 50:
            results.append(
                {"status": "warning", "message": f"Small dataset ({len(X_train)} samples). Results might be unstable."}
            )
        else:
            results.append({"status": "success", "message": f"Training data has {len(X_train)} samples."})

        if problem_type == "Classification":
            try:
                unique_targets = len(np.unique(y_train.dropna()))
            except Exception:
                unique_targets = 0
            if unique_targets < 2:
                results.append(
                    {"status": "error", "message": "Classification requires at least 2 unique target classes."}
                )
            else:
                results.append({"status": "success", "message": f"Found {unique_targets} classes for classification."})

        if "Linear" in model_type or "Logistic" in model_type:
            results.append(
                {
                    "status": "info",
                    "message": f"{model_type} assumes feature scaling. Check if scaling was applied.",
                }
            )
        if "Gradient" in model_type and X_train.isnull().sum().sum() > 0:
            results.append(
                {"status": "warning", "message": f"{model_type} might be sensitive to missing values."}
            )

        return results
