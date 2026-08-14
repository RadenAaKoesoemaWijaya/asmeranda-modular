"""
Pydantic schemas untuk request/response API.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DatasetMetadata(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    numerical_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str] = []
    size_bytes: int
    uploaded_at: str


class DatasetListResponse(BaseModel):
    datasets: List[DatasetMetadata]
    total: int


class DatasetUploadResponse(BaseModel):
    success: bool
    metadata: Optional[DatasetMetadata] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
class EdaSummaryResponse(BaseModel):
    success: bool
    metadata: Optional[Dict[str, Any]] = None
    shape: Optional[Dict[str, int]] = None
    dtypes: Optional[Dict[str, str]] = None
    describe_numeric: Optional[Dict[str, Any]] = None
    describe_categorical: Optional[Dict[str, Any]] = None
    missing: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class EdaCorrelationResponse(BaseModel):
    success: bool
    columns: Optional[List[str]] = None
    matrix: Optional[List[List[float]]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
class PreprocessingConfig(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None
    problem_type: Optional[str] = Field(default=None, pattern="^(Classification|Regression|Forecasting)$")
    numerical_features: Optional[List[str]] = None
    categorical_features: Optional[List[str]] = None
    scaling_method: str = "auto"  # auto|standard|minmax|robust|power|quantile
    imputation_strategy: str = "auto"  # auto|mean|median|most_frequent|drop
    apply_polynomial: bool = False
    apply_binning: bool = False
    apply_encoding: bool = True
    test_size: float = 0.2
    random_state: int = 42


class PreprocessingResponse(BaseModel):
    success: bool
    state_id: Optional[str] = None
    n_samples_train: Optional[int] = None
    n_samples_test: Optional[int] = None
    n_features: Optional[int] = None
    feature_names: Optional[List[str]] = None
    target_column: Optional[str] = None
    problem_type: Optional[str] = None
    preprocessing_steps: Optional[List[str]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class TrainingConfig(BaseModel):
    state_id: str
    model_type: str = "RandomForest"
    problem_type: str = "Classification"
    hyperparams: Optional[Dict[str, Any]] = None
    cv_method: str = "kfold"  # none|kfold|stratified|loo|timeseries
    cv_folds: int = 5
    random_state: int = 42


class TrainingResponse(BaseModel):
    success: bool
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    cv_scores: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None  # For background task status messages
