"""Model training, evaluation, tracking, and persistence modules."""

from coffee_quality.models.baseline import (
    CrossValidationResult,
    build_dummy_pipeline,
    build_logistic_pipeline,
    evaluate_cv,
    train_and_evaluate_baseline,
)
from coffee_quality.models.evaluator import EvaluationMetrics, evaluate_predictions
from coffee_quality.models.registry import (
    get_client,
    init_mlflow,
    load_model_by_run_id,
    load_registered_model,
    log_experiment_run,
    promote_model_alias,
)

__all__ = [
    "CrossValidationResult",
    "EvaluationMetrics",
    "build_dummy_pipeline",
    "build_logistic_pipeline",
    "evaluate_cv",
    "evaluate_predictions",
    "get_client",
    "init_mlflow",
    "load_model_by_run_id",
    "load_registered_model",
    "log_experiment_run",
    "promote_model_alias",
    "train_and_evaluate_baseline",
]
