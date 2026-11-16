"""Monitoring route handlers for data and target drift detection."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import HTMLResponse

from coffee_quality.api.audit import AuditLogger, get_audit_logger
from coffee_quality.api.dependencies import get_settings_dependency
from coffee_quality.config import Settings
from coffee_quality.monitoring.drift import DriftDetector

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get(
    "/drift",
    status_code=status.HTTP_200_OK,
    summary="Data and Target Drift Summary",
    description="Evaluate statistical drift between recent live inference samples and the reference baseline.",
)
async def get_drift_summary(
    limit: int = Query(
        default=500,
        ge=5,
        le=10000,
        description="Maximum number of recent inference audit records to evaluate.",
    ),
    audit_logger: AuditLogger = Depends(get_audit_logger),
    settings: Settings = Depends(get_settings_dependency),
) -> dict[str, Any]:
    """Calculate and return structured statistical drift metrics for active production data.

    Args:
        limit: Number of recent audit records to include in evaluation.
        audit_logger: Structured audit logging service.
        settings: Application settings.

    Returns:
        Dictionary containing overall drift status, drifted feature share, and per-feature metrics.
    """
    logger.info("Computing drift summary on recent live inference data (limit=%d)...", limit)
    current_df = audit_logger.to_dataframe(limit=limit)

    detector = DriftDetector.from_settings(settings=settings)
    summary, _ = detector.calculate_drift(current_df)

    return summary.to_dict()


@router.get(
    "/drift/html",
    response_class=HTMLResponse,
    status_code=status.HTTP_200_OK,
    summary="Interactive EvidentlyAI Drift Report",
    description="Render a standalone HTML dashboard report visualizing distribution comparisons and drift tests.",
)
async def get_drift_html_report(
    limit: int = Query(
        default=500,
        ge=5,
        le=10000,
        description="Maximum number of recent inference audit records to evaluate.",
    ),
    audit_logger: AuditLogger = Depends(get_audit_logger),
    settings: Settings = Depends(get_settings_dependency),
) -> HTMLResponse:
    """Render interactive HTML drift report for visual dashboard embedding.

    Args:
        limit: Number of recent audit records to include in evaluation.
        audit_logger: Structured audit logging service.
        settings: Application settings.

    Returns:
        HTMLResponse containing standalone Evidently report markup.
    """
    logger.info("Rendering interactive HTML drift report (limit=%d)...", limit)
    current_df = audit_logger.to_dataframe(limit=limit)

    detector = DriftDetector.from_settings(settings=settings)
    _, html_report = detector.calculate_drift(current_df)

    return HTMLResponse(content=html_report, status_code=status.HTTP_200_OK)
