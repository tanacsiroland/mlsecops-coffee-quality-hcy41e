"""Model training, evaluation, and persistence modules."""

from coffee_quality.models.baseline import (
    CrossValidationResult,
    build_dummy_pipeline,
    build_logistic_pipeline,
    evaluate_cv,
    train_and_evaluate_baseline,
)
from coffee_quality.models.evaluator import EvaluationMetrics, evaluate_predictions

__all__ = [
    "CrossValidationResult",
    "EvaluationMetrics",
    "build_dummy_pipeline",
    "build_logistic_pipeline",
    "evaluate_cv",
    "evaluate_predictions",
    "train_and_evaluate_baseline",
]
