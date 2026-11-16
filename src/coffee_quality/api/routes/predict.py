"""Prediction route handlers for the coffee quality classification service."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Any, cast

from fastapi import APIRouter, Depends, status
import numpy as np

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
    metadata: ModelMetadata = Depends(get_model_metadata),
) -> PredictionResponse:
    """Classify a batch of coffee samples as Specialty (1) or Commercial (0).

    Args:
        payload: Validated batch of coffee sample attributes.
        metadata: Active model pipeline and version metadata.

    Returns:
        PredictionResponse containing individual predictions, latency, and model metadata.
    """
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

    logger.info(
        "Processed batch prediction for %d samples using model %s in %.2f ms",
        len(predictions),
        metadata.version,
        latency_ms,
    )

    return PredictionResponse(
        predictions=predictions,
        model_version=metadata.version,
        count=len(predictions),
        inference_latency_ms=latency_ms,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
