"""API route handlers and routers."""

from coffee_quality.api.routes.predict import router as predict_router
from coffee_quality.api.routes.train import router as train_router

__all__ = ["predict_router", "train_router"]
