"""FastAPI dependency injection utilities for settings and verified model loading."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import cast

import cloudpickle
from fastapi import Depends, HTTPException, Request, status
from sklearn.pipeline import Pipeline

from coffee_quality.config import Settings, get_settings
from coffee_quality.models.registry import get_client, load_registered_model
from coffee_quality.security.integrity import ModelIntegrityError, safe_load_model

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMetadata:
    """Operational container holding an active in-memory model and its metadata."""

    model: Pipeline
    version: str
    alias: str
    loaded_at_utc: str


class ModelContainer:
    """Thread-safe in-memory cache supporting zero-downtime model hot-reloading."""

    def __init__(self) -> None:
        self._current: ModelMetadata | None = None
        self._lock = threading.Lock()

    def get(self) -> ModelMetadata | None:
        """Return the currently cached model metadata, or None if empty."""
        with self._lock:
            return self._current

    def set(self, model: Pipeline, version: str, alias: str = "champion") -> ModelMetadata:
        """Atomically update the active in-memory model cache."""
        with self._lock:
            meta = ModelMetadata(
                model=model,
                version=version,
                alias=alias,
                loaded_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            self._current = meta
            return meta

    def clear(self) -> None:
        """Clear cached model reference from memory."""
        with self._lock:
            self._current = None


global_model_container = ModelContainer()


def get_settings_dependency(request: Request) -> Settings:
    """Inject application settings from request app state or environment fallback."""
    settings = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return get_settings()


def load_production_model(settings: Settings | None = None) -> ModelMetadata:
    """Resolve and load the active champion model following the security hierarchy.

    1. Primary: Local signed artifact file verified via SHA-256 cryptographic manifest.
    2. Secondary: MLflow Model Registry using the 'champion' alias.
    3. Fallback: Raises HTTP 503 if no model is available.

    Args:
        settings: Application settings containing artifact paths and MLflow URIs.

    Returns:
        ModelMetadata containing verified Pipeline and version info.

    Raises:
        HTTPException(500): If cryptographic artifact integrity verification fails.
        HTTPException(503): If no model is available from any source.
    """
    cfg = settings or get_settings()
    artifacts_dir: Path = cfg.model_artifacts_dir
    model_path = artifacts_dir / "champion.bin"
    manifest_path = artifacts_dir / "champion.manifest.json"

    # 1. Attempt loading signed local artifact if both binary and manifest exist
    if model_path.is_file() and manifest_path.is_file():
        logger.info("Found local model artifact at %s. Verifying integrity...", model_path)
        try:
            if cfg.enable_artifact_integrity_check:
                loaded = safe_load_model(model_path, manifest_path)
            else:
                with open(model_path, "rb") as f:
                    loaded = cloudpickle.load(f)

            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            version = "local_champion"
            if isinstance(manifest_data, dict):
                version = str(manifest_data.get("metadata", {}).get("version", "local_champion"))

            logger.info("Successfully loaded verified local model version: %s", version)
            return global_model_container.set(cast(Pipeline, loaded), version=version, alias="champion")
        except ModelIntegrityError as exc:
            logger.error("Model cryptographic integrity check failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model artifact cryptographic integrity check failed: {exc}",
            ) from exc

    # 2. Attempt loading from MLflow Model Registry via 'champion' alias
    try:
        logger.info("Attempting to load 'champion' model from MLflow registry at %s...", cfg.mlflow_tracking_uri)
        pipeline = load_registered_model(
            model_name="coffee-quality-model",
            alias="champion",
            tracking_uri=cfg.mlflow_tracking_uri,
        )

        client = get_client(tracking_uri=cfg.mlflow_tracking_uri)
        mv = client.get_model_version_by_alias("coffee-quality-model", "champion")
        version = f"v{mv.version}"

        logger.info("Successfully loaded registered model version %s from MLflow.", version)
        return global_model_container.set(pipeline, version=version, alias="champion")
    except Exception as exc:
        logger.warning("Could not load model from MLflow registry: %s", exc)

    # 3. Fallback: No model available
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Machine learning model is not currently available. Please trigger /train to train a champion model.",
    )


def get_model_metadata(
    request: Request,
    settings: Settings = Depends(get_settings_dependency),
) -> ModelMetadata:
    """Dependency returning the active ModelMetadata container, loading if necessary."""
    # Check app state first
    app_model = getattr(request.app.state, "model", None)
    app_alias = getattr(request.app.state, "model_alias", "champion")
    if isinstance(app_model, Pipeline):
        return ModelMetadata(
            model=app_model,
            version=str(getattr(request.app.state, "model_version", "app_state")),
            alias=str(app_alias),
            loaded_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    # Check global thread-safe container
    cached = global_model_container.get()
    if cached is not None:
        request.app.state.model = cached.model
        request.app.state.model_alias = cached.alias
        request.app.state.model_version = cached.version
        return cached

    # Load from storage or registry
    meta = load_production_model(settings)
    request.app.state.model = meta.model
    request.app.state.model_alias = meta.alias
    request.app.state.model_version = meta.version
    return meta


def get_model_pipeline(
    metadata: ModelMetadata = Depends(get_model_metadata),
) -> Pipeline:
    """FastAPI dependency injecting the verified scikit-learn Pipeline into route handlers."""
    return metadata.model
