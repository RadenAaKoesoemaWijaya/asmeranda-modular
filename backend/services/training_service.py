"""
Training service - latih model sklearn/xgboost/lightgbm/catboost dan simpan.

Service ini tidak langsung melatih semua model (itu di training
endpoint) - hanya factory + persist. Model disimpan ke
``settings.data_dir/models/{model_id}.pkl``.
"""
from __future__ import annotations

import logging
import pickle
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from backend.core.config import settings
try:
    from backend.core.model_security import sign_model_file, verify_model_integrity
except ImportError:
    from core.model_security import sign_model_file, verify_model_integrity

logger = logging.getLogger("asmeranda.services.training")


# Optional: framework tambahan
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore

    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False


_MODEL_DIR = Path(settings.data_dir) / "models"
_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# In-memory model metadata
_MODELS: Dict[str, Dict[str, Any]] = {}


def _load_models_from_disk() -> None:
    """Muat ulang metadata model dari file .pkl yang tersimpan."""
    if not _MODEL_DIR.exists():
        return
    for model_path in sorted(_MODEL_DIR.glob("*.pkl")):
        model_id = model_path.stem
        if model_id in _MODELS:
            continue
        is_valid, _ = verify_model_integrity(model_path, allow_legacy_unsigned=True)
        if not is_valid:
            continue
        try:
            with open(model_path, "rb") as fh:
                payload = pickle.load(fh)
        except Exception:
            continue
        _MODELS[model_id] = {
            "model_id": model_id,
            "model_type": payload.get("model_type"),
            "problem_type": payload.get("problem_type"),
            "metrics": None,
            "cv_report": None,
            "path": str(model_path),
            "n_features": len(payload.get("feature_names") or []),
            "feature_names": payload.get("feature_names") or [],
        }


_load_models_from_disk()


def _build_voting_ensemble(model_type: str, problem_type: str, hyperparams: Optional[Dict[str, Any]] = None):
    """Build voting ensemble with default base estimators."""
    hp = hyperparams or {}
    is_clf = problem_type == "Classification"
    
    # Default base estimators
    if is_clf:
        estimators = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
            ('gb', GradientBoostingClassifier(random_state=42, n_estimators=100)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]
        voting = hp.get('voting', 'soft')  # 'hard' or 'soft'
        return VotingClassifier(estimators=estimators, voting=voting)
    else:
        estimators = [
            ('rf', RandomForestRegressor(random_state=42, n_estimators=100)),
            ('gb', GradientBoostingRegressor(random_state=42, n_estimators=100)),
            ('lr', LinearRegression())
        ]
        return VotingRegressor(estimators=estimators)


def _build_stacking_ensemble(model_type: str, problem_type: str, hyperparams: Optional[Dict[str, Any]] = None):
    """Build stacking ensemble with default base estimators."""
    hp = hyperparams or {}
    is_clf = problem_type == "Classification"
    
    # Default base estimators
    if is_clf:
        estimators = [
            ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
            ('gb', GradientBoostingClassifier(random_state=42, n_estimators=100)),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)
        return StackingClassifier(estimators=estimators, final_estimator=final_estimator)
    else:
        estimators = [
            ('rf', RandomForestRegressor(random_state=42, n_estimators=100)),
            ('gb', GradientBoostingRegressor(random_state=42, n_estimators=100)),
            ('lr', LinearRegression())
        ]
        final_estimator = LinearRegression()
        return StackingRegressor(estimators=estimators, final_estimator=final_estimator)


def _build_model(model_type: str, problem_type: str, hyperparams: Optional[Dict[str, Any]] = None):
    """Factory model sesuai tipe."""
    hp = (hyperparams or {}).copy()
    is_clf = problem_type == "Classification"

    mt = model_type.lower().replace("-", "_").replace(" ", "_")
    if mt in ("random_forest", "randomforest", "rf", "random_forst"):
        return (RandomForestClassifier if is_clf else RandomForestRegressor)(random_state=42, **hp)
    if mt in ("gradient_boosting", "gradientboosting", "gb", "gbm"):
        return (GradientBoostingClassifier if is_clf else GradientBoostingRegressor)(random_state=42, **hp)
    if mt in ("logistic", "logistic_regression", "logisticregression", "lr"):
        return LogisticRegression(max_iter=1000, **hp) if is_clf else LinearRegression(**hp)
    if mt in ("linear", "linear_regression", "linearregression"):
        return LinearRegression(**hp)
    if mt in ("decision_tree", "decisiontree", "dt"):
        return (DecisionTreeClassifier if is_clf else DecisionTreeRegressor)(random_state=42, **hp)
    if mt in ("knn", "kneighbors", "k_neighbors"):
        return (KNeighborsClassifier if is_clf else KNeighborsRegressor)(**hp)
    if mt in ("svm", "svc", "svr", "support_vector_machine"):
        if is_clf:
            return SVC(probability=True, **hp)
        else:
            return SVR(**hp)
    if mt in ("xgboost", "xgb") and XGBOOST_AVAILABLE:
        return (XGBClassifier if is_clf else XGBRegressor)(random_state=42, **hp)
    if mt in ("lightgbm", "lgbm", "light_gbm") and LIGHTGBM_AVAILABLE:
        return (LGBMClassifier if is_clf else LGBMRegressor)(random_state=42, **hp)
    if mt in ("catboost", "cat_boost", "cb") and CATBOOST_AVAILABLE:
        return (CatBoostClassifier if is_clf else CatBoostRegressor)(random_state=42, verbose=False, **hp)
    if mt in ("voting", "votingclassifier", "votingregressor", "voting_ensemble"):
        return _build_voting_ensemble(model_type, problem_type, hp)
    if mt in ("stacking", "stackingclassifier", "stackingregressor", "stacking_ensemble"):
        return _build_stacking_ensemble(model_type, problem_type, hp)
    raise ValueError(f"Model type '{model_type}' tidak dikenali atau library tidak terpasang.")


def _make_cv(cv_method: str, n_splits: int, y=None):
    if cv_method == "none":
        return None
    if cv_method == "stratified":
        if y is None or y.nunique() < 2:
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)
        min_class_count = int(y.value_counts().min())
        if min_class_count < 2:
            return KFold(n_splits=n_splits, shuffle=True, random_state=42)
        # Adapt n_splits if minority class count is smaller than n_splits
        effective_splits = max(2, min(n_splits, min_class_count))
        return StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)
    if cv_method == "loo":
        return LeaveOneOut()
    if cv_method == "timeseries":
        return TimeSeriesSplit(n_splits=n_splits)
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def _score_classification(y_true, y_pred, y_proba=None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }
    try:
        if y_proba is not None and len(np.unique(y_true)) == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    except Exception:
        pass
    return out


def _score_regression(y_true, y_pred) -> Dict[str, Any]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str,
    problem_type: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    cv_method: str = "kfold",
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Latih model, kembalikan model_id, metrics, dan cv_scores.

    Model disimpan ke ``settings.data_dir/models/{model_id}.pkl``.
    """
    if X_train is None or y_train is None:
        return {"success": False, "error": "X_train/y_train tidak boleh None"}
    if X_test is None or y_test is None:
        return {"success": False, "error": "X_test/y_test tidak boleh None"}

    # Pastikan semua kolom numerik
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    try:
        model = _build_model(model_type, problem_type, hyperparams)
    except Exception as exc:
        return {"success": False, "error": f"Gagal membuat model: {exc}"}

    # Fit
    try:
        model.fit(X_train, y_train)
    except Exception as exc:
        return {"success": False, "error": f"Gagal melatih model: {exc}"}

    # Predict
    try:
        y_pred = model.predict(X_test)
    except Exception as exc:
        return {"success": False, "error": f"Gagal memprediksi: {exc}"}

    y_proba = None
    if problem_type == "Classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    if problem_type == "Classification":
        metrics = _score_classification(y_test, y_pred, y_proba)
    else:
        metrics = _score_regression(y_test, y_pred)

    # Cross-validation (optional)
    cv_report: Optional[Dict[str, Any]] = None
    cv = _make_cv(cv_method, cv_folds, y_train)
    if cv is not None and problem_type == "Classification":
        scoring = "accuracy"
    elif cv is not None and problem_type == "Regression":
        scoring = "r2"
    else:
        scoring = None
    if cv is not None and scoring is not None:
        try:
            from sklearn.base import clone
            cv_model = clone(model)
            # Use all available CPU cores for parallel cross-validation
            scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            cv_report = {
                "method": cv_method,
                "folds": cv_folds,
                "scoring": scoring,
                "scores": scores.tolist(),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
        except Exception as exc:
            cv_report = {"method": cv_method, "error": str(exc)}

    # Feature Importances (if available)
    feature_importances = []
    if hasattr(model, "feature_importances_"):
        try:
            importances = model.feature_importances_.tolist()
            feature_importances = [
                {"feature": str(col), "importance": float(imp)}
                for col, imp in zip(X_train.columns, importances)
            ]
            feature_importances.sort(key=lambda x: x["importance"], reverse=True)
        except Exception:
            pass
    elif hasattr(model, "coef_"):
        try:
            coefs = np.abs(model.coef_)
            if coefs.ndim > 1:
                coefs = coefs.mean(axis=0)
            coefs_list = coefs.tolist()
            feature_importances = [
                {"feature": str(col), "importance": float(imp)}
                for col, imp in zip(X_train.columns, coefs_list)
            ]
            feature_importances.sort(key=lambda x: x["importance"], reverse=True)
        except Exception:
            pass

    # Save model
    model_id = uuid.uuid4().hex
    path = _MODEL_DIR / f"{model_id}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "model": model,
                "model_type": model_type,
                "problem_type": problem_type,
                "feature_names": list(X_train.columns),
                "hyperparams": hyperparams or {},
                "metrics": metrics,
                "cv_report": cv_report,
                "feature_importances": feature_importances,
            },
            fh,
        )

    try:
        sign_model_file(path, metadata={"model_id": model_id, "model_type": model_type, "problem_type": problem_type})
    except Exception as exc:
        logger.warning(f"Could not create cryptographic signature for model {model_id}: {exc}")

    _MODELS[model_id] = {
        "model_id": model_id,
        "model_type": model_type,
        "problem_type": problem_type,
        "metrics": metrics,
        "cv_report": cv_report,
        "feature_importances": feature_importances,
        "path": str(path),
        "n_features": int(X_train.shape[1]),
        "feature_names": list(X_train.columns),
    }

    return {
        "success": True,
        "model_id": model_id,
        "metrics": metrics,
        "cv_scores": cv_report,
        "feature_importances": feature_importances,
        "model_type": model_type,
        "problem_type": problem_type,
    }



def list_models() -> Dict[str, Dict[str, Any]]:
    return _MODELS


def load_model(model_id: str):
    path = _MODEL_DIR / f"{model_id}.pkl"
    if not path.exists():
        return None
    is_valid, msg = verify_model_integrity(path, allow_legacy_unsigned=True)
    if not is_valid:
        logger.error(f"Cannot load tampered/untrusted model {model_id}: {msg}")
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def get_metadata(model_id: str) -> Optional[Dict[str, Any]]:
    return _MODELS.get(model_id)


def save_uploaded_model(file_bytes: bytes, original_filename: str = "uploaded_model.pkl") -> Dict[str, Any]:
    """Parse and save an uploaded model (.pkl) to disk and memory registry."""
    import io
    from sklearn.base import is_classifier, is_regressor

    try:
        obj = pickle.loads(file_bytes)
    except Exception as exc:
        return {"success": False, "error": f"Format file .pkl tidak valid atau corrupt: {str(exc)}"}

    model_id = f"custom_{uuid.uuid4().hex[:8]}"

    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        feature_names = obj.get("feature_names") or []
        problem_type = obj.get("problem_type")
        model_type = obj.get("model_type") or type(model).__name__
        metrics = obj.get("metrics")
        cv_report = obj.get("cv_report")
        feature_importances = obj.get("feature_importances")
    else:
        model = obj
        model_type = type(model).__name__
        if is_classifier(model) or hasattr(model, "predict_proba"):
            problem_type = "Classification"
        elif is_regressor(model):
            problem_type = "Regression"
        else:
            problem_type = "Classification"
        feature_names = getattr(model, "feature_names_in_", None)
        if feature_names is not None:
            feature_names = list(feature_names)
        else:
            feature_names = []
        metrics = None
        cv_report = None
        feature_importances = None

    # Persist in standard dictionary format
    path = _MODEL_DIR / f"{model_id}.pkl"
    payload = {
        "model": model,
        "model_type": model_type,
        "problem_type": problem_type,
        "feature_names": feature_names,
        "metrics": metrics,
        "cv_report": cv_report,
        "feature_importances": feature_importances,
        "original_filename": original_filename,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)

    try:
        sign_model_file(path, metadata={"model_id": model_id, "model_type": model_type, "original_filename": original_filename})
    except Exception as exc:
        logger.warning(f"Could not sign uploaded model {model_id}: {exc}")

    _MODELS[model_id] = {
        "model_id": model_id,
        "model_type": model_type,
        "problem_type": problem_type,
        "metrics": metrics,
        "cv_report": cv_report,
        "feature_importances": feature_importances,
        "path": str(path),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "original_filename": original_filename,
    }

    return {
        "success": True,
        "model_id": model_id,
        "model_type": model_type,
        "problem_type": problem_type,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "message": f"Model '{model_type}' berhasil diimpor dengan ID {model_id}",
    }


def delete_model(model_id: str) -> bool:
    path = _MODEL_DIR / f"{model_id}.pkl"
    sig_path = _MODEL_DIR / f"{model_id}.pkl.sig"
    existed = False
    if path.exists():
        try:
            path.unlink()
        except Exception:
            pass
        existed = True
    if sig_path.exists():
        try:
            sig_path.unlink()
        except Exception:
            pass
    _MODELS.pop(model_id, None)
    return existed


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    model_types: Optional[List[str]] = None,
    cv_method: str = "kfold",
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Compare multiple models and return performance comparison.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    problem_type : str
        'Classification' or 'Regression'
    model_types : List[str]
        List of model types to compare
    cv_method : str
        Cross-validation method
    cv_folds : int
        Number of CV folds
        
    Returns
    -------
    dict
        Comparison results with metrics for each model
    """
    if model_types is None:
        if problem_type == "Classification":
            model_types = ["RandomForest", "GradientBoosting", "LogisticRegression", "DecisionTree", "KNeighbors"]
        else:
            model_types = ["RandomForest", "GradientBoosting", "LinearRegression", "DecisionTree", "KNeighbors"]
    
    results = []
    
    for model_type in model_types:
        try:
            # Train model
            result = train(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_type=model_type,
                problem_type=problem_type,
                cv_method=cv_method,
                cv_folds=cv_folds,
            )
            
            if result["success"]:
                model_info = {
                    "model_type": model_type,
                    "model_id": result["model_id"],
                    "metrics": result["metrics"],
                    "cv_scores": result["cv_scores"],
                }
                results.append(model_info)
            else:
                results.append({
                    "model_type": model_type,
                    "error": result.get("error", "Training failed")
                })
        except Exception as e:
            results.append({
                "model_type": model_type,
                "error": str(e)
            })
    
    # Rank models by performance
    if problem_type == "Classification":
        ranking_key = "accuracy"
    else:
        ranking_key = "r2"
    
    valid_results = [r for r in results if "metrics" in r and ranking_key in r["metrics"]]
    ranked_results = sorted(
        valid_results,
        key=lambda x: x["metrics"][ranking_key],
        reverse=(problem_type == "Classification")
    )
    
    return {
        "success": True,
        "problem_type": problem_type,
        "results": results,
        "ranking": ranked_results,
        "best_model": ranked_results[0] if ranked_results else None,
        "ranking_metric": ranking_key
    }
