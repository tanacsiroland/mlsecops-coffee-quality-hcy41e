"""FastAPI REST service, schemas, dependencies, and route endpoints."""

from coffee_quality.api.app import app, create_app
from coffee_quality.api.dependencies import (
    ModelContainer,
    ModelMetadata,
    get_model_metadata,
    get_model_pipeline,
    get_settings_dependency,
    global_model_container,
    load_production_model,
)
from coffee_quality.api.schemas import (
    CoffeeBatchRequest,
    CoffeePredictionInput,
    PredictionItem,
    PredictionResponse,
    TrainingTriggerRequest,
    TrainingTriggerResponse,
)

__all__ = [
    "CoffeeBatchRequest",
    "CoffeePredictionInput",
    "ModelContainer",
    "ModelMetadata",
    "PredictionItem",
    "PredictionResponse",
    "TrainingTriggerRequest",
    "TrainingTriggerResponse",
    "app",
    "create_app",
    "get_model_metadata",
    "get_model_pipeline",
    "get_settings_dependency",
    "global_model_container",
    "load_production_model",
]
