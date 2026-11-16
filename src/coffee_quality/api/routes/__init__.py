"""API route handlers and routers."""

from coffee_quality.api.routes.monitoring import router as monitoring_router
from coffee_quality.api.routes.predict import router as predict_router
from coffee_quality.api.routes.train import router as train_router

__all__ = ["monitoring_router", "predict_router", "train_router"]
