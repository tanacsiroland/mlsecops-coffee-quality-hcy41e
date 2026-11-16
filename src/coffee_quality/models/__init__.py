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
from coffee_quality.models.trainer import (
    build_competitive_pipeline,
    run_training_pipeline,
    train_and_evaluate_pipeline,
)

__all__ = [
    "CrossValidationResult",
    "EvaluationMetrics",
    "build_competitive_pipeline",
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
    "run_training_pipeline",
    "train_and_evaluate_baseline",
    "train_and_evaluate_pipeline",
]
