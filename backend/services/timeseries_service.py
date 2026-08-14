"""
Timeseries service - forecasting & anomaly detection sederhana.

Menggunakan:
- ``statsmodels`` untuk stationarity test & seasonality decomposition
- ``sklearn`` IsolationForest untuk anomaly detection
- ``utils.prepare_timeseries_data`` & ``utils.advanced_data_scaling`` (refactored)

Catatan: tidak semua algoritma berat (Prophet, DLinear) dipakai sebagai
default; fokus pada yang cepat & robust. Frontend bisa menambahkan
pilihan algoritma tambahan nanti.
"""
from __future__ import annotations

import io
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import adfuller  # type: ignore

    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest  # type: ignore

    ISO_FOREST_AVAILABLE = True
except Exception:
    ISO_FOREST_AVAILABLE = False

from backend.services import dataset_service


def _detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Coba deteksi kolom datetime pertama."""
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].head(20), errors="raise")
                return col
            except Exception:
                continue
    return None


def detect_timeseries(
    dataset_id: str,
    target_column: Optional[str] = None,
    date_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Analisis awal data timeseries: stasioneritas, seasonality, outliers."""
    df = dataset_service.get_dataset(dataset_id)
    if df is None:
        return {"success": False, "error": f"Dataset {dataset_id} tidak ditemukan"}

    if date_column is None:
        date_column = _detect_datetime_column(df)
    if target_column is None:
        # Pilih kolom numerik dengan unique values paling banyak
        num = df.select_dtypes(include=["number"])
        if num.shape[1] == 0:
            return {"success": False, "error": "Tidak ada kolom numerik"}
        target_column = num.var().idxmax()

    if target_column not in df.columns:
        return {"success": False, "error": f"Kolom target {target_column} tidak ada"}
    if date_column and date_column not in df.columns:
        return {"success": False, "error": f"Kolom date {date_column} tidak ada"}

    series = df[target_column].dropna()
    if len(series) < 10:
        return {"success": False, "error": "Data terlalu sedikit (min 10 baris)"}

    result: Dict[str, Any] = {
        "success": True,
        "dataset_id": dataset_id,
        "target_column": target_column,
        "date_column": date_column,
        "n_observations": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "is_stationary": None,
        "adf_statistic": None,
        "adf_pvalue": None,
        "n_anomalies": 0,
        "anomaly_indices": [],
    }

    # ADF test
    if STATSMODELS_AVAILABLE:
        try:
            adf_result = adfuller(series.values, autolag="AIC")
            result["adf_statistic"] = float(adf_result[0])
            result["adf_pvalue"] = float(adf_result[1])
            result["is_stationary"] = bool(adf_result[1] < 0.05)
        except Exception as exc:
            result["adf_error"] = str(exc)

    # IsolationForest anomaly detection
    if ISO_FOREST_AVAILABLE:
        try:
            iso = IsolationForest(contamination=0.05, random_state=42)
            arr = series.values.reshape(-1, 1)
            preds = iso.fit_predict(arr)
            anomaly_idx = np.where(preds == -1)[0].tolist()
            result["n_anomalies"] = int(len(anomaly_idx))
            result["anomaly_indices"] = [int(i) for i in anomaly_idx[:200]]  # cap
        except Exception as exc:
            result["anomaly_error"] = str(exc)

    return result


def forecast(
    dataset_id: str,
    target_column: str,
    date_column: Optional[str] = None,
    horizon: int = 10,
    method: str = "naive",  # naive | drift | mean
) -> Dict[str, Any]:
    """
    Forecasting sederhana (naive / drift / mean) untuk smoke-test.
    Forecast yang lebih akurat (Prophet/ARIMA) bisa ditambahkan nanti.
    """
    df = dataset_service.get_dataset(dataset_id)
    if df is None:
        return {"success": False, "error": f"Dataset {dataset_id} tidak ditemukan"}
    if target_column not in df.columns:
        return {"success": False, "error": f"Kolom target {target_column} tidak ada"}
    series = df[target_column].dropna().reset_index(drop=True)
    if len(series) < horizon + 2:
        return {"success": False, "error": "Data terlalu sedikit untuk forecasting"}

    horizon = int(max(1, min(horizon, len(series))))
    last_idx = len(series) - 1

    forecast_values: List[float] = []
    if method == "naive":
        forecast_values = [float(series.iloc[-1])] * horizon
    elif method == "drift":
        # Linear trend
        first = float(series.iloc[0])
        last = float(series.iloc[-1])
        slope = (last - first) / max(last_idx, 1)
        forecast_values = [float(last + slope * (i + 1)) for i in range(horizon)]
    elif method == "mean":
        forecast_values = [float(series.mean())] * horizon
    else:
        return {"success": False, "error": f"Method {method} tidak dikenal. Gunakan naive|drift|mean."}

    return {
        "success": True,
        "method": method,
        "target_column": target_column,
        "horizon": horizon,
        "last_observed": float(series.iloc[-1]),
        "forecast": forecast_values,
        "forecast_index": [int(last_idx + i + 1) for i in range(horizon)],
    }


def anomaly_detection(
    dataset_id: str,
    target_column: str,
    contamination: float = 0.05,
) -> Dict[str, Any]:
    """Deteksi anomali pada satu kolom numerik."""
    if not ISO_FOREST_AVAILABLE:
        return {"success": False, "error": "IsolationForest tidak tersedia"}
    df = dataset_service.get_dataset(dataset_id)
    if df is None:
        return {"success": False, "error": f"Dataset {dataset_id} tidak ditemukan"}
    if target_column not in df.columns:
        return {"success": False, "error": f"Kolom target {target_column} tidak ada"}
    series = df[target_column].dropna().reset_index(drop=True)
    if len(series) < 10:
        return {"success": False, "error": "Data terlalu sedikit"}

    arr = series.values.reshape(-1, 1)
    iso = IsolationForest(contamination=float(contamination), random_state=42)
    preds = iso.fit_predict(arr)
    scores = iso.decision_function(arr)
    anomaly_idx = np.where(preds == -1)[0]
    anomaly_records = [
        {
            "index": int(i),
            "value": float(series.iloc[i]),
            "score": float(scores[i]),
        }
        for i in anomaly_idx
    ]
    return {
        "success": True,
        "target_column": target_column,
        "n_observations": int(len(series)),
        "n_anomalies": int(len(anomaly_idx)),
        "contamination": float(contamination),
        "anomalies": anomaly_records,
    }
