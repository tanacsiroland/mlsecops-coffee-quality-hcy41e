"""Model retraining route handlers for the coffee quality classification service."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Request, status

from coffee_quality.api.dependencies import (
    get_settings_dependency,
    load_production_model,
)
from coffee_quality.api.schemas import (
    TrainingTriggerRequest,
    TrainingTriggerResponse,
)
from coffee_quality.config import Settings
from coffee_quality.models.trainer import run_training_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Training"])

# Global concurrency lock preventing concurrent training pipeline execution
training_lock = threading.Lock()


@router.post(
    "/train",
    response_model=TrainingTriggerResponse,
    status_code=status.HTTP_200_OK,
    summary="Trigger Model Pipeline Retraining",
    description="Execute the complete training pipeline, evaluate against test split, and promote champion model.",
)
async def trigger_training(
    request: Request,
    payload: TrainingTriggerRequest | None = None,
    settings: Settings = Depends(get_settings_dependency),
) -> TrainingTriggerResponse:
    """Execute model training pipeline, log metrics, and optionally hot-reload champion model.

    Args:
        request: Incoming FastAPI HTTP request.
        payload: Optional configuration specifying model architecture and champion promotion.
        settings: Application settings containing dataset paths and MLflow URIs.

    Returns:
        TrainingTriggerResponse with run ID, evaluation metrics, and promotion status.

    Raises:
        HTTPException(409): If another retraining pipeline is currently running.
        HTTPException(500): If pipeline execution encounters an unrecoverable failure.
    """
    req_payload = payload if payload is not None else TrainingTriggerRequest()

    if not training_lock.acquire(blocking=False):
        logger.warning("Rejected concurrent /train invocation: training lock is currently held.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A model retraining pipeline is currently running. Please wait for completion.",
        )

    try:
        logger.info(
            "Initiating training pipeline (model_type=%s, register_as_champion=%s)...",
            req_payload.model_type,
            req_payload.register_as_champion,
        )

        run_id, comp_test_metrics, _ = run_training_pipeline(
            raw_data_path=settings.data_raw_path,
            model_type=req_payload.model_type,
            random_state=settings.random_state,
            register_as_champion=req_payload.register_as_champion,
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        )

        # Zero-downtime hot-reload of active in-memory champion model
        if req_payload.register_as_champion:
            try:
                meta = load_production_model(settings)
                request.app.state.model = meta.model
                request.app.state.model_version = meta.version
                request.app.state.model_alias = meta.alias
                logger.info("Successfully hot-reloaded active champion model: %s", meta.version)
            except Exception as reload_err:
                logger.warning("Retraining succeeded but hot-reloading encountered error: %s", reload_err)

        metrics_dict: dict[str, float] = {
            "accuracy": round(float(comp_test_metrics.accuracy), 4),
            "macro_f1": round(float(comp_test_metrics.macro_f1), 4),
            "precision_specialty": round(float(comp_test_metrics.precision_specialty), 4),
            "recall_specialty": round(float(comp_test_metrics.recall_specialty), 4),
            "f1_specialty": round(float(comp_test_metrics.f1_specialty), 4),
        }
        if comp_test_metrics.roc_auc is not None:
            metrics_dict["roc_auc"] = round(float(comp_test_metrics.roc_auc), 4)

        logger.info(
            "Training pipeline completed successfully for run %s with Macro-F1: %.4f",
            run_id,
            comp_test_metrics.macro_f1,
        )

        return TrainingTriggerResponse(
            status="completed",
            run_id=run_id,
            model_type=req_payload.model_type,
            promoted_as_champion=req_payload.register_as_champion,
            metrics=metrics_dict,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:
        logger.exception("Model retraining execution failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model retraining execution failed: {exc}",
        ) from exc
    finally:
        training_lock.release()
