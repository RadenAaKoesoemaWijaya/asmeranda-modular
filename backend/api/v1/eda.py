"""
Endpoint /eda - Exploratory Data Analysis (summary, correlation, dll).
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import polars as pl
from fastapi import APIRouter, HTTPException, Query

from backend.schemas.models import EdaCorrelationResponse, EdaSummaryResponse
from backend.services import dataset_service

logger = logging.getLogger("asmeranda.api.eda")
router = APIRouter()


@router.get("/{dataset_id}/summary", response_model=EdaSummaryResponse)
def summary(dataset_id: str) -> EdaSummaryResponse:
    """Ringkasan dataset: shape, dtypes, describe, missing values."""
    try:
        result = dataset_service.summary(dataset_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} tidak ditemukan")
        return EdaSummaryResponse(success=True, **result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("EDA summary gagal")
        return EdaSummaryResponse(success=False, error=str(exc))


@router.get("/{dataset_id}/correlation", response_model=EdaCorrelationResponse)
def correlation(
    dataset_id: str,
    columns: str = Query(default="", description="Comma-separated column names; kosong = semua numerik"),
) -> EdaCorrelationResponse:
    """Matriks korelasi Pearson antar kolom numerik."""
    try:
        df = dataset_service.get_dataset_pl(dataset_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} tidak ditemukan")

        if columns.strip():
            cols = [c.strip() for c in columns.split(",") if c.strip()]
            num_df = df.select(cols).select(pl.col(pl.NUMERIC_DTYPES))
        else:
            num_df = df.select(pl.col(pl.NUMERIC_DTYPES))

        if num_df.width < 2:
            return EdaCorrelationResponse(
                success=False,
                error="Butuh minimal 2 kolom numerik untuk korelasi.",
            )

        # Polars pearson correlation matrix
        corr = num_df.pearson_corr()
        # Konversi ke pandas hanya untuk operasi replace/fillna JSON safety yang mudah
        # atau gunakan Polars fill_nan / fill_null
        corr_pd = corr.to_pandas().fillna(0.0).replace([np.inf, -np.inf], 0.0)
        matrix = corr_pd.round(4).values.tolist()
        return EdaCorrelationResponse(
            success=True,
            columns=num_df.columns.tolist(),
            matrix=matrix,
        )
    except HTTPException:
        raise
    except Exception as exc:
        return EdaCorrelationResponse(success=False, error=str(exc))


@router.get("/{dataset_id}/data")
def paginated_data(
    dataset_id: str,
    page: int = Query(1, ge=1, description="Nomor halaman"),
    size: int = Query(50, ge=1, le=1000, description="Jumlah baris per halaman"),
):
    """Ambil sebagian raw data (server-side pagination)"""
    try:
        result = dataset_service.get_paginated_data(dataset_id, page, size)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} tidak ditemukan")
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Paginated data gagal")
        return {"success": False, "error": str(exc)}
