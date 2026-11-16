"""Prediction route handlers for the coffee quality classification service."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Any, cast
import uuid

from fastapi import APIRouter, Depends, Request, Response, status
import numpy as np

from coffee_quality.api.audit import AuditLogger, get_audit_logger
from coffee_quality.api.dependencies import ModelMetadata, get_model_metadata
from coffee_quality.api.schemas import (
    CoffeeBatchRequest,
    PredictionItem,
    PredictionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Coffee Quality Classification",
    description="Score a batch of Arabica coffee lot attributes to predict specialty grade status.",
)
async def predict_coffee_quality(
    payload: CoffeeBatchRequest,
    request: Request,
    response: Response,
    metadata: ModelMetadata = Depends(get_model_metadata),
    audit_logger: AuditLogger = Depends(get_audit_logger),
) -> PredictionResponse:
    """Classify a batch of coffee samples as Specialty (1) or Commercial (0).

    Args:
        payload: Validated batch of coffee sample attributes.
        request: Incoming FastAPI HTTP request.
        response: Outgoing FastAPI HTTP response.
        metadata: Active model pipeline and version metadata.
        audit_logger: Structured audit logging service.

    Returns:
        PredictionResponse containing individual predictions, latency, and model metadata.
    """
    request_id = str(uuid.uuid4())
    response.headers["X-Request-ID"] = request_id

    start_time = time.perf_counter()
    df = payload.to_dataframe()

    model = metadata.model

    if hasattr(model, "predict_proba"):
        probs = cast(np.ndarray[Any, Any], model.predict_proba(df))
        if probs.shape[1] > 1:
            specialty_probs = [float(p) for p in probs[:, 1]]
        else:
            specialty_probs = [float(p) for p in probs[:, 0]]
    else:
        preds = model.predict(df)
        specialty_probs = [1.0 if int(p) == 1 else 0.0 for p in preds]

    predictions = [PredictionItem.from_probability(prob) for prob in specialty_probs]

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    # Log structured inference event to audit trail
    client_ip = request.client.host if request.client else None
    audit_logger.log(
        request_id=request_id,
        model_version=metadata.version,
        model_alias=metadata.alias,
        item_count=len(predictions),
        inference_latency_ms=latency_ms,
        inputs=df.to_dict(orient="records"),
        predictions=[p.model_dump() for p in predictions],
        client_ip=client_ip,
        timestamp_utc=timestamp_utc,
    )

    logger.info(
        "Processed batch prediction for %d samples using model %s in %.2f ms [request_id: %s]",
        len(predictions),
        metadata.version,
        latency_ms,
        request_id,
    )

    return PredictionResponse(
        predictions=predictions,
        model_version=metadata.version,
        count=len(predictions),
        inference_latency_ms=latency_ms,
        timestamp_utc=timestamp_utc,
    )
