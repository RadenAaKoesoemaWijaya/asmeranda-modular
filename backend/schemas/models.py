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
class FeatureSelectionConfig(BaseModel):
    method: str = "none"  # none|variance|correlation|kbest|rfe
    max_features: int = 10
    threshold: float = 0.05


class ImbalanceConfig(BaseModel):
    method: str = "none"  # none|oversample|undersample|smote|adasyn
    sampling_strategy: str = "auto"


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
    feature_selection: Optional[FeatureSelectionConfig] = None
    imbalance_handling: Optional[ImbalanceConfig] = None
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
    feature_selection_info: Optional[Dict[str, Any]] = None
    imbalance_handling_info: Optional[Dict[str, Any]] = None
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


class EvaluationConfig(BaseModel):
    state_id: str
    model_id: str
    generate_plots: bool = True
    plot_types: List[str] = ["confusion_matrix", "roc_curve", "feature_importance"]


class TrainingResponse(BaseModel):
    success: bool
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    cv_scores: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    message: Optional[str] = None  # For background task status messages


class EvaluationResponse(BaseModel):
    success: bool
    model_id: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    plots: Optional[Dict[str, str]] = None  # plot_type -> base64_encoded_image
    error: Optional[str] = None


class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]  # Input data for prediction


class PredictionResponse(BaseModel):
    success: bool
    predictions: Optional[List[Any]] = None
    probabilities: Optional[List[List[float]]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
class ClusteringConfig(BaseModel):
    state_id: str
    method: str = "kmeans"  # kmeans, dbscan, hierarchical, spectral
    parameters: Dict[str, Any] = {}


class ClusteringResponse(BaseModel):
    success: bool
    labels: Optional[List[int]] = None
    metrics: Optional[Dict[str, Any]] = None
    method: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
class OptimizationConfig(BaseModel):
    state_id: str
    model_type: str = "RandomForest"
    problem_type: str = "Classification"
    method: str = "grid_search"  # grid_search, random_search, bayesian
    cv_folds: int = 5
    n_iter: int = 50  # for random_search and bayesian


class OptimizationResponse(BaseModel):
    success: bool
    best_params: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    method: Optional[str] = None
    cv_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------
class RecommendationRequest(BaseModel):
    dataset_id: str


class RecommendationResponse(BaseModel):
    success: bool
    recommendations: Optional[List[Dict[str, Any]]] = None
    dataset_info: Optional[Dict[str, Any]] = None
    preprocessing_steps: Optional[List[str]] = None
    error: Optional[str] = None
