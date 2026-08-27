"""
Advanced ML API endpoints.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services.advanced_ml_service import AdvancedMLService
from backend.services.utilities_service import UtilitiesService
from core.state import get_state

logger = logging.getLogger("asmeranda.api.advanced_ml")
router = APIRouter()
advanced_ml_service = AdvancedMLService()
utilities_service = UtilitiesService()


def _get_dataframe_and_targets(state_id: str) -> Tuple[Optional[pd.DataFrame], Optional[List[Any]], Optional[str]]:
    """Helper to extract DataFrame and targets from state registry."""
    state = get_state(state_id)
    if not state:
        return None, None, f"State ID '{state_id}' tidak ditemukan. Selesaikan preprocessing terlebih dahulu."

    df = None
    # Check possible keys in state
    for key in ["X_train", "data", "training_data", "df"]:
        if key in state and state[key] is not None:
            val = state[key]
            if isinstance(val, pd.DataFrame):
                df = val.copy()
                break
            elif isinstance(val, list) and len(val) > 0:
                df = pd.DataFrame(val)
                break
            elif isinstance(val, np.ndarray):
                df = pd.DataFrame(val)
                break

    if df is None or df.empty:
        return None, None, "Tidak ada data training yang tersedia. Jalankan Preprocessing terlebih dahulu."

    # Extract target if present
    targets = None
    y_val = state.get("y_train")
    if y_val is not None:
        if isinstance(y_val, (pd.Series, np.ndarray, list)):
            targets = list(y_val)

    return df, targets, None


# ── Request Models ────────────────────────────────────────────────────────
class UMAPRequest(BaseModel):
    state_id: str
    method: str = "umap"  # umap | pca | tsne
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"


class HDBSCANRequest(BaseModel):
    state_id: str
    min_cluster_size: int = 5
    min_samples: Optional[int] = None
    metric: str = "euclidean"


class AnomalyDetectionRequest(BaseModel):
    state_id: str
    method: str = "isolation_forest"  # isolation_forest | one_class_svm
    contamination: float = 0.1
    n_estimators: int = 100


class ForecastingRequest(BaseModel):
    state_id: str
    target_column: str
    periods: int = 10
    method: str = "arima"  # arima | exp_smoothing | moving_avg | linear | simple


class MissingValueRequest(BaseModel):
    state_id: str
    strategy: str = "auto"
    numeric_strategy: str = "mean"
    categorical_strategy: str = "mode"
    threshold: float = 0.5


class OutlierDetectionRequest(BaseModel):
    state_id: str
    method: str = "iqr"
    threshold: float = 1.5
    columns: Optional[List[str]] = None


class DataValidationRequest(BaseModel):
    state_id: str
    required_columns: Optional[List[str]] = None
    column_types: Optional[Dict[str, str]] = None
    value_ranges: Optional[Dict[str, Any]] = None


# ── Endpoints ─────────────────────────────────────────────────────────────
@router.post("/umap")
def umap_dimensionality_reduction(request: UMAPRequest) -> Dict[str, Any]:
    """Perform UMAP / PCA / t-SNE dimensionality reduction."""
    try:
        data, targets, err = _get_dataframe_and_targets(request.state_id)
        if err:
            return {"success": False, "error": err}

        if request.method == "pca":
            result = advanced_ml_service.pca_dimensionality_reduction(
                data=data,
                n_components=request.n_components,
                targets=targets,
            )
        elif request.method == "tsne":
            result = advanced_ml_service.tsne_dimensionality_reduction(
                data=data,
                n_components=request.n_components,
                targets=targets,
            )
        else:
            result = advanced_ml_service.umap_dimensionality_reduction(
                data=data,
                n_components=request.n_components,
                n_neighbors=request.n_neighbors,
                min_dist=request.min_dist,
                metric=request.metric,
                targets=targets,
            )

        return result

    except Exception as e:
        logger.error(f"Dimensionality reduction failed: {e}")
        return {"success": False, "error": str(e), "method": request.method}


@router.post("/hdbscan")
def hdbscan_clustering(request: HDBSCANRequest) -> Dict[str, Any]:
    """Perform HDBSCAN clustering with metrics and 2D visual projection."""
    try:
        data, _, err = _get_dataframe_and_targets(request.state_id)
        if err:
            return {"success": False, "error": err}

        result = advanced_ml_service.hdbscan_clustering(
            data=data,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples,
            metric=request.metric,
        )
        return result

    except Exception as e:
        logger.error(f"HDBSCAN clustering failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/anomaly-detection")
def anomaly_detection(request: AnomalyDetectionRequest) -> Dict[str, Any]:
    """Perform anomaly detection using Isolation Forest or One-Class SVM."""
    try:
        data, _, err = _get_dataframe_and_targets(request.state_id)
        if err:
            return {"success": False, "error": err}

        if request.method == "one_class_svm":
            result = advanced_ml_service.one_class_svm_anomaly_detection(
                data=data,
                nu=request.contamination,
                kernel="rbf",
            )
        else:
            result = advanced_ml_service.isolation_forest_anomaly_detection(
                data=data,
                contamination=request.contamination,
                n_estimators=request.n_estimators,
            )
        return result

    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/forecast")
def forecast(request: ForecastingRequest) -> Dict[str, Any]:
    """Perform time series forecasting."""
    try:
        data, _, err = _get_dataframe_and_targets(request.state_id)
        if err:
            return {"success": False, "error": err}

        # Check if target column is in data or original dataset
        if request.target_column not in data.columns:
            state = get_state(request.state_id)
            if state and "data" in state and isinstance(state["data"], pd.DataFrame):
                if request.target_column in state["data"].columns:
                    data = state["data"]

        if request.target_column not in data.columns:
            return {
                "success": False,
                "error": f"Kolom '{request.target_column}' tidak ditemukan dalam dataset. Kolom tersedia: {list(data.columns)[:10]}",
            }

        result = advanced_ml_service.basic_forecasting(
            data=data,
            target_column=request.target_column,
            periods=request.periods,
            method=request.method,
        )
        return result

    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/handle-missing-values")
def handle_missing_values(request: MissingValueRequest) -> Dict[str, Any]:
    """Handle missing values in the dataset."""
    try:
        data, _, err = _get_dataframe_and_targets(request.state_id)
        if err:
            return {"success": False, "error": err}

        result = utilities_service.handle_missing_values(
            data=data,
            strategy=request.strategy,
            numeric_strategy=request.numeric_strategy,
            categorical_strategy=request.categorical_strategy,
            threshold=request.threshold,
        )

        if result.get("success"):
            state = get_state(request.state_id)
            if state:
                state["X_train"] = result["data"]
            result["original_shape"] = list(result["original_shape"])
            result["new_shape"] = list(result["new_shape"])
            result["data"] = {
                "shape": result["new_shape"],
                "processed": True,
                "preview": result["data"].head(10).to_dict(orient="records"),
            }

        return result

    except Exception as e:
        logger.error(f"Missing value handling failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/detect-outliers")
def detect_outliers(request: OutlierDetectionRequest) -> Dict[str, Any]:
    """Detect outliers in the dataset."""
    try:
        data, _, err = _get_dataframe_and_targets(request.state_id)
        if err:
            return {"success": False, "error": err}

        result = utilities_service.detect_outliers(
            data=data,
            method=request.method,
            threshold=request.threshold,
            columns=request.columns,
        )
        return result

    except Exception as e:
        logger.error(f"Outlier detection failed: {e}")
        return {"success": False, "error": str(e)}


@router.post("/validate-data")
def validate_data(request: DataValidationRequest) -> Dict[str, Any]:
    """Validate data against constraints."""
    try:
        data, _, err = _get_dataframe_and_targets(request.state_id)
        if err:
            return {"success": False, "error": err}

        result = utilities_service.validate_data(
            data=data,
            required_columns=request.required_columns,
            column_types=request.column_types,
            value_ranges=request.value_ranges,
        )
        return {"success": True, "data": result}

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return {"success": False, "error": str(e)}


@router.get("/columns/{state_id}")
def get_columns(state_id: str) -> Dict[str, Any]:
    """Get available columns and types for a state."""
    try:
        data, _, err = _get_dataframe_and_targets(state_id)
        if err:
            return {"success": False, "error": err}

        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
        return {
            "success": True,
            "columns": list(data.columns),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "n_rows": len(data),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}