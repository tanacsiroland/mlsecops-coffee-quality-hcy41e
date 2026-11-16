"""FastAPI application factory, lifecycle management, and core endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Any, AsyncGenerator, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from coffee_quality.api.routes.monitoring import router as monitoring_router
from coffee_quality.api.routes.predict import router as predict_router
from coffee_quality.api.routes.train import router as train_router
from coffee_quality.config import Settings, get_settings

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """System health and operational status response payload."""

    status: str = Field(default="healthy", description="Operational health status string")
    app_env: str = Field(description="Active execution environment")
    api_version: str = Field(description="Semantic version of the serving service")
    model_loaded: bool = Field(description="Whether a machine learning model is loaded in memory")
    model_alias: str | None = Field(default=None, description="Active model registry alias if loaded")
    timestamp_utc: str = Field(description="UTC timestamp of the health check")


class RootResponse(BaseModel):
    """Root service information and navigation payload."""

    service: str = Field(description="Service display name")
    version: str = Field(description="Semantic version")
    docs_url: str = Field(default="/docs", description="Interactive OpenAPI documentation path")
    health_url: str = Field(default="/health", description="System health status path")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI application lifecycle management context."""
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    logger.info("Initializing %s in '%s' environment...", settings.api_title, settings.app_env)
    yield
    logger.info("Shutting down %s...", settings.api_title)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure a production-ready FastAPI application instance.

    Args:
        settings: Optional Settings override for testing or isolated environments.

    Returns:
        Configured FastAPI application instance.
    """
    app_settings = settings or get_settings()

    app = FastAPI(
        title=app_settings.api_title,
        version=app_settings.api_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.state.settings = app_settings
    app.state.model = None
    app.state.model_alias = None

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OWASP Security Headers Middleware
    @app.middleware("http")
    async def add_security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    # Global unhandled exception handler: sanitize error messages
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception encountered on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error occurred."},
        )

    # Core routes
    @app.get(
        "/",
        response_model=RootResponse,
        summary="Service Discovery Root",
        tags=["System"],
    )
    async def root() -> dict[str, Any]:
        """Return basic service metadata and navigation endpoints."""
        return {
            "service": app_settings.api_title,
            "version": app_settings.api_version,
            "docs_url": "/docs",
            "health_url": "/health",
        }

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="System Health & Readiness Check",
        tags=["System"],
    )
    async def health() -> HealthResponse:
        """Report server health, deployment environment, and active model status."""
        model_is_loaded = getattr(app.state, "model", None) is not None
        active_alias = getattr(app.state, "model_alias", None)

        return HealthResponse(
            status="healthy",
            app_env=app_settings.app_env,
            api_version=app_settings.api_version,
            model_loaded=model_is_loaded,
            model_alias=active_alias,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    # Route modules
    app.include_router(predict_router)
    app.include_router(train_router)
    app.include_router(monitoring_router)

    return app


# Default singleton app instance for Uvicorn
app = create_app()
