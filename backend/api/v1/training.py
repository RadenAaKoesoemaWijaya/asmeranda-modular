"""
Endpoint /training - latih model dari state hasil preprocessing.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.schemas.models import TrainingConfig, TrainingResponse
from backend.services import training_service
from core.state import get_state

logger = logging.getLogger("asmeranda.api.training")
router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


def _train_model_background(
    state_id: str,
    X_train,
    y_train,
    X_test,
    y_test,
    model_type: str,
    problem_type: str,
    hyperparams: Dict[str, Any],
    cv_method: str,
    cv_folds: int,
) -> Dict[str, Any]:
    """Background task for model training to avoid blocking requests."""
    try:
        result = training_service.train(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_type=model_type,
            problem_type=problem_type,
            hyperparams=hyperparams,
            cv_method=cv_method,
            cv_folds=cv_folds,
        )
        
        if result.get("success"):
            logger.info(
                "Background model training completed successfully",
                extra={
                    "model_id": result.get("model_id"),
                    "model_type": model_type,
                    "problem_type": problem_type,
                    "state_id": state_id
                }
            )
        else:
            logger.error(
                "Background model training failed",
                extra={
                    "error": result.get("error"),
                    "model_type": model_type,
                    "problem_type": problem_type,
                    "state_id": state_id
                }
            )
        
        return result
    except Exception as exc:
        logger.error(
            "Background training error",
            exc_info=True,
            extra={
                "state_id": state_id,
                "model_type": model_type,
                "error_type": type(exc).__name__
            }
        )
        return {"success": False, "error": str(exc)}


@router.post("/start", response_model=TrainingResponse)
@limiter.limit("5/minute")  # Limit to 5 training requests per minute per IP
def start_training(request: Request, background_tasks: BackgroundTasks, config: TrainingConfig) -> TrainingResponse:
    """Latih model berdasarkan state hasil preprocessing (async background task)."""
    try:
        state = get_state(config.state_id)
        X_train = state.get("X_train")
        X_test = state.get("X_test")
        y_train = state.get("y_train")
        y_test = state.get("y_test")

        if X_train is None or y_train is None or X_test is None or y_test is None:
            logger.warning(
                "Invalid preprocessing state",
                extra={
                    "state_id": config.state_id,
                    "model_type": config.model_type,
                    "problem_type": config.problem_type
                }
            )
            raise HTTPException(
                status_code=400,
                detail="State preprocessing tidak valid (X_train/X_test/y_train/y_test hilang).",
            )

        # Add training as background task to avoid blocking
        background_tasks.add_task(
            _train_model_background,
            config.state_id,
            X_train,
            y_train,
            X_test,
            y_test,
            config.model_type,
            config.problem_type,
            config.hyperparams or {},
            config.cv_method,
            config.cv_folds,
        )
        
        # Return immediate response indicating training started
        logger.info(
            "Model training started in background",
            extra={
                "state_id": config.state_id,
                "model_type": config.model_type,
                "problem_type": config.problem_type
            }
        )
        
        return TrainingResponse(
            success=True,
            model_id="pending",  # Will be updated when background task completes
            message="Training started in background. Check /models endpoint for results."
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Unexpected error during training start",
            exc_info=True,
            extra={
                "state_id": config.state_id,
                "model_type": config.model_type,
                "error_type": type(exc).__name__
            }
        )
        raise HTTPException(status_code=500, detail=f"Training start error: {str(exc)}")


@router.get("/models")
def list_models() -> Dict[str, Any]:
    """List semua model yang sudah dilatih."""
    return training_service.list_models()


@router.get("/models/{model_id}")
def get_model(model_id: str) -> Dict[str, Any]:
    meta = training_service.get_metadata(model_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} tidak ditemukan")
    return meta


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    ok = training_service.delete_model(model_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Model {model_id} tidak ditemukan")
    return {"success": True, "model_id": model_id, "deleted": True}
