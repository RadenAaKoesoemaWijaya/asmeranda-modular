"""
Endpoint /datasets - upload, list, get, delete dataset tabular.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.schemas.models import (
    DatasetListResponse,
    DatasetMetadata,
    DatasetUploadResponse,
)
from backend.services import dataset_service
from backend.core.config import settings

logger = logging.getLogger("asmeranda.api.datasets")
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("", response_model=DatasetListResponse)
def list_datasets() -> DatasetListResponse:
    """List semua dataset yang sudah di-upload."""
    items = dataset_service.list_datasets()
    return DatasetListResponse(datasets=items, total=len(items))


@router.post("", response_model=DatasetUploadResponse)
@limiter.limit("10/minute")  # Limit to 10 uploads per minute per IP
async def upload_dataset(request: Request, file: UploadFile = File(...)) -> DatasetUploadResponse:
    """
    Upload file dataset (CSV/XLSX/Parquet/JSON/TSV).
    File disimpan ke ``settings.data_dir/{dataset_id}.parquet``.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="File kosong")
        
        # Validate file size before processing
        file_size_mb = len(content) / (1024 * 1024)
        max_size_mb = settings.max_upload_size_mb
        if file_size_mb > max_size_mb:
            raise HTTPException(
                status_code=413, 
                detail=f"File terlalu besar ({file_size_mb:.2f}MB). Maksimum {max_size_mb}MB"
            )
        
        meta = dataset_service.ingest(
            content=content,
            filename=file.filename or "dataset",
            original_name=file.filename,
        )
        return DatasetUploadResponse(success=True, metadata=meta)
    except ValueError as exc:
        logger.warning(
            "Upload validation failed: %s",
            str(exc),
            extra={"dataset_filename": file.filename},
        )
        return DatasetUploadResponse(success=False, error=str(exc))
    except Exception as exc:
        logger.error(
            "Upload dataset failed unexpectedly",
            exc_info=True,
            extra={
                "dataset_filename": file.filename,
                "file_size_mb": len(content) / (1024 * 1024) if content else 0,
                "error_type": type(exc).__name__,
            },
        )
        return DatasetUploadResponse(success=False, error=f"Internal error: {exc}")


@router.get("/{dataset_id}", response_model=DatasetMetadata)
def get_dataset(dataset_id: str) -> DatasetMetadata:
    """Ambil metadata dataset (tidak termasuk isi)."""
    meta = dataset_service.get_metadata(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} tidak ditemukan")
    return DatasetMetadata(**meta)


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: str):
    """Hapus dataset."""
    ok = dataset_service.delete_dataset(dataset_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} tidak ditemukan")
    return {"success": True, "dataset_id": dataset_id, "deleted": True}
