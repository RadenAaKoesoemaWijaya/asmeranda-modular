"""
Preprocessing service - handling imputasi, scaling, encoding, train-test split.

Service ini pure-Python (tidak depend on FastAPI). Output disimpan
ke state registry (``core.state``) dan state_id dikembalikan ke caller.
"""
from __future__ import annotations

import pickle
import uuid
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from core.state import get_state


def _impute(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    strategy: str,
) -> pd.DataFrame:
    """Isi missing values (mean/median/most_frequent/drop/auto)."""
    out = df.copy()
    if strategy in ("auto", ""):
        strategy = "mean"
    if strategy == "drop":
        out = out.dropna().reset_index(drop=True)
        return out
    for c in num_cols:
        if c in out.columns and out[c].isnull().any():
            if strategy == "median":
                out[c] = out[c].fillna(out[c].median())
            elif strategy == "most_frequent":
                mode_val = out[c].mode()
                fill = mode_val.iloc[0] if len(mode_val) else 0
                out[c] = out[c].fillna(fill)
            else:  # default: mean
                out[c] = out[c].fillna(out[c].mean())
    for c in cat_cols:
        if c in out.columns and out[c].isnull().any():
            mode_val = out[c].mode()
            fill = mode_val.iloc[0] if len(mode_val) else "missing"
            out[c] = out[c].fillna(fill)
    return out


def _scale(
    X: pd.DataFrame,
    num_cols: List[str],
    method: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Scaling kolom numerik sesuai metode."""
    info: Dict[str, Any] = {"method": method}
    if method == "none" or not num_cols:
        return X, info
    valid_cols = [c for c in num_cols if c in X.columns]
    if not valid_cols:
        return X, info
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "power":
        scaler = PowerTransformer(method="yeo-johnson")
    elif method == "quantile":
        scaler = QuantileTransformer(output_distribution="normal", random_state=0)
    else:
        # auto: pilih otomatis
        skew = X[valid_cols].skew().abs().mean()
        if skew > 1.0:
            scaler = PowerTransformer(method="yeo-johnson")
            method_use = "power"
        elif skew > 0.5:
            scaler = QuantileTransformer(output_distribution="normal", random_state=0)
            method_use = "quantile"
        else:
            scaler = StandardScaler()
            method_use = "standard"
        info["method"] = method_use
        scaler.fit(X[valid_cols])
        X[valid_cols] = scaler.transform(X[valid_cols])
        info["scaler"] = pickle.dumps(scaler).hex()
        return X, info

    scaler.fit(X[valid_cols])
    X[valid_cols] = scaler.transform(X[valid_cols])
    info["scaler"] = pickle.dumps(scaler).hex()
    return X, info


def _encode(
    X: pd.DataFrame,
    cat_cols: List[str],
    apply_encoding: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """One-hot encoding untuk kolom kategorikal (dibatasi agar ukuran tidak meledak)."""
    info: Dict[str, Any] = {"encoded_columns": []}
    if not apply_encoding or not cat_cols:
        return X, info
    valid_cols = [c for c in cat_cols if c in X.columns]
    for c in valid_cols:
        n_unique = X[c].nunique(dropna=True)
        if n_unique <= 20:
            # One-hot encode
            dummies = pd.get_dummies(X[c], prefix=c, drop_first=True, dummy_na=False)
            X = pd.concat([X.drop(columns=[c]), dummies], axis=1)
            info["encoded_columns"].extend(dummies.columns.tolist())
        else:
            # Frequency encoding (lebih aman untuk high-cardinality)
            freq = X[c].value_counts(normalize=True)
            X[c + "_freq"] = X[c].map(freq)
            X = X.drop(columns=[c])
            info["encoded_columns"].append(c + "_freq")
    return X, info


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Jalankan preprocessing sesuai config.

    Parameters
    ----------
    config : dict
        {
          'dataset_id': str,
          'target_column': str|None,
          'problem_type': 'Classification'|'Regression'|'Forecasting'|None,
          'numerical_features': [str],
          'categorical_features': [str],
          'scaling_method': 'auto'|'standard'|...,
          'imputation_strategy': 'auto'|'mean'|...,
          'apply_encoding': bool,
          'test_size': float,
          'random_state': int,
        }

    Returns
    -------
    dict
        {
          'success': bool,
          'state_id': str,
          'n_samples_train': int,
          'n_samples_test': int,
          'n_features': int,
          'feature_names': [str],
          'target_column': str|None,
          'problem_type': str|None,
          'preprocessing_steps': [str],
          'error': str|None,
        }
    """
    from backend.services import dataset_service
    # Import WebSocket manager (lazy, agar tidak circular import)
    try:
        from backend.api.v1.ws import manager as ws_manager
    except Exception:
        ws_manager = None

    async def _broadcast(d_id: str, progress: int, message: str):
        if ws_manager and d_id:
            await ws_manager.broadcast(d_id, {"progress": progress, "message": message})

    def _try_broadcast(d_id: str, progress: int, message: str):
        """Coba broadcast tanpa blocking."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_broadcast(d_id, progress, message))
            else:
                loop.run_until_complete(_broadcast(d_id, progress, message))
        except Exception:
            pass  # broadcast gagal tidak boleh menghentikan preprocessing

    dataset_id = config.get("dataset_id")
    df = dataset_service.get_dataset(dataset_id) if dataset_id else None
    if df is None:
        return {"success": False, "error": f"Dataset {dataset_id} tidak ditemukan"}

    _try_broadcast(dataset_id, 5, "Memulai preprocessing...")
    steps: List[str] = []
    target_column = config.get("target_column")
    problem_type = config.get("problem_type")

    # Tentukan kolom fitur
    if config.get("numerical_features") is None:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    else:
        num_cols = list(config["numerical_features"])
    if config.get("categorical_features") is None:
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    else:
        cat_cols = list(config["categorical_features"])

    if target_column and target_column in num_cols:
        num_cols = [c for c in num_cols if c != target_column]
    if target_column and target_column in cat_cols:
        cat_cols = [c for c in cat_cols if c != target_column]

    # 1) Imputasi
    df_clean = _impute(df, num_cols, cat_cols, config.get("imputation_strategy", "auto"))
    steps.append(f"imputation={config.get('imputation_strategy', 'auto')}")
    _try_broadcast(dataset_id, 30, "Imputasi selesai. Memisahkan target...")

    # 2) Pisahkan target
    y: Optional[pd.Series] = None
    if target_column and target_column in df_clean.columns:
        y = df_clean[target_column]
        X = df_clean.drop(columns=[target_column])
    else:
        X = df_clean.copy()

    # 3) Scaling numerik
    X, scale_info = _scale(X, num_cols, config.get("scaling_method", "auto"))
    steps.append(f"scaling={scale_info.get('method', 'auto')}")
    _try_broadcast(dataset_id, 60, "Scaling selesai. Encoding kategorikal...")

    # 4) Encoding kategorikal
    X, enc_info = _encode(X, cat_cols, config.get("apply_encoding", True))
    if enc_info.get("encoded_columns"):
        steps.append(f"encoding={len(enc_info['encoded_columns'])} new cols")
    _try_broadcast(dataset_id, 80, "Encoding selesai. Train-test split...")

    # 5) Train-test split
    test_size = float(config.get("test_size", 0.2))
    random_state = int(config.get("random_state", 42))
    if y is not None and problem_type in ("Classification", "Regression"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if problem_type == "Classification" and y.nunique() > 1 else None,
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    else:
        # Forecasting / unsupervised: split sequential
        n = len(X)
        idx = int(n * (1 - test_size))
        X_train, X_test = X.iloc[:idx], X.iloc[idx:]
        y_train = y.iloc[:idx] if y is not None else None
        y_test = y.iloc[idx:] if y is not None else None

    steps.append(f"split={1 - test_size:.2f}/{test_size:.2f}")
    _try_broadcast(dataset_id, 95, "Split selesai. Menyimpan state...")

    # 6) Simpan ke state registry
    state_id = uuid.uuid4().hex
    state = get_state(state_id)
    state["data"] = df
    state["target_column"] = target_column
    state["problem_type"] = problem_type
    state["X_train"] = X_train
    state["X_test"] = X_test
    state["y_train"] = y_train
    state["y_test"] = y_test
    state["numerical_columns"] = [c for c in num_cols if c in X.columns]
    state["categorical_columns"] = [c for c in cat_cols if c in X.columns]
    state["feature_names"] = X.columns.tolist()
    state["scaler_info"] = scale_info
    state["encoding_info"] = enc_info

    _try_broadcast(dataset_id, 100, "Preprocessing selesai!")
    return {
        "success": True,
        "state_id": state_id,
        "n_samples_train": int(len(X_train)),
        "n_samples_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "feature_names": X_train.columns.tolist(),
        "target_column": target_column,
        "problem_type": problem_type,
        "preprocessing_steps": steps,
    }
