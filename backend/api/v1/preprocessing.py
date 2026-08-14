"""
Endpoint /preprocessing - jalankan preprocessing dan simpan ke state.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.schemas.models import PreprocessingConfig, PreprocessingResponse
from backend.services import preprocessing_service

logger = logging.getLogger("asmeranda.api.preprocessing")
router = APIRouter()


@router.post("/run", response_model=PreprocessingResponse)
def run_preprocessing(config: PreprocessingConfig) -> PreprocessingResponse:
    """Jalankan preprocessing sesuai konfigurasi."""
    try:
        result = preprocessing_service.run(config.dict())
        if not result.get("success"):
            return PreprocessingResponse(success=False, error=result.get("error"))
        return PreprocessingResponse(
            success=True,
            state_id=result["state_id"],
            n_samples_train=result["n_samples_train"],
            n_samples_test=result["n_samples_test"],
            n_features=result["n_features"],
            feature_names=result["feature_names"],
            target_column=result["target_column"],
            problem_type=result["problem_type"],
            preprocessing_steps=result["preprocessing_steps"],
        )
    except Exception as exc:
        logger.exception("Preprocessing gagal")
        return PreprocessingResponse(success=False, error=str(exc))
