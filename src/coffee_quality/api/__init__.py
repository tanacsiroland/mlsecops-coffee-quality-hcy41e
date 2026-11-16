"""FastAPI REST service, schemas, dependencies, and route endpoints."""

from coffee_quality.api.app import app, create_app
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
    "PredictionItem",
    "PredictionResponse",
    "TrainingTriggerRequest",
    "TrainingTriggerResponse",
    "app",
    "create_app",
]
