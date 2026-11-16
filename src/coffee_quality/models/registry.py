"""MLflow experiment tracking and model registry integration."""

from __future__ import annotations

from typing import Any, cast

import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from coffee_quality.config import get_settings
from coffee_quality.models.baseline import CrossValidationResult
from coffee_quality.models.evaluator import EvaluationMetrics


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    """Return an MLflowClient bound to the configured tracking URI.

    Args:
        tracking_uri: Optional tracking URI override. Falls back to application settings.

    Returns:
        Configured MlflowClient instance.
    """
    uri = tracking_uri or get_settings().mlflow_tracking_uri
    return MlflowClient(tracking_uri=uri)


def init_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> str:
    """Initialize MLflow tracking URI and activate the target experiment.

    Args:
        tracking_uri: Optional tracking URI override (e.g. 'sqlite:///mlflow.db').
        experiment_name: Optional experiment name override.

    Returns:
        The active experiment ID string.
    """
    settings = get_settings()
    uri = tracking_uri or settings.mlflow_tracking_uri
    exp_name = experiment_name or settings.mlflow_experiment_name

    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        exp_id = mlflow.create_experiment(exp_name)
    else:
        exp_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name=exp_name)
    return exp_id


def log_experiment_run(
    pipeline: Pipeline,
    run_name: str,
    params: dict[str, Any],
    cv_results: CrossValidationResult,
    test_metrics: EvaluationMetrics,
    tags: dict[str, str] | None = None,
    registered_model_name: str | None = None,
) -> str:
    """Log a complete training run to MLflow including params, metrics, artifacts, and pipeline.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        run_name: Human-readable name for this run.
        params: Model hyperparameters dictionary.
        cv_results: Cross-validation aggregated metrics across folds.
        test_metrics: Holdout test set evaluation metrics.
        tags: Optional custom audit and metadata tags.
        registered_model_name: Optional model name to register in the Model Registry.

    Returns:
        The generated MLflow run ID.
    """
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # 1. Hyperparameters
        mlflow.log_params(params)

        # 2. Cross-validation metrics (prefixed with 'cv_')
        cv_dict = {f"cv_{k}": v for k, v in cv_results.to_dict().items()}
        mlflow.log_metrics(cv_dict)

        # 3. Holdout test metrics (prefixed with 'test_')
        test_dict: dict[str, float] = {
            "test_accuracy": test_metrics.accuracy,
            "test_macro_f1": test_metrics.macro_f1,
            "test_precision_commercial": test_metrics.precision_commercial,
            "test_recall_commercial": test_metrics.recall_commercial,
            "test_f1_commercial": test_metrics.f1_commercial,
            "test_precision_specialty": test_metrics.precision_specialty,
            "test_recall_specialty": test_metrics.recall_specialty,
            "test_f1_specialty": test_metrics.f1_specialty,
        }
        if test_metrics.roc_auc is not None:
            test_dict["test_roc_auc"] = test_metrics.roc_auc
        mlflow.log_metrics(test_dict)

        # 4. Diagnostic artifacts (confusion matrix json)
        cm_artifact = {
            "confusion_matrix": test_metrics.confusion_matrix,
            "true_negatives": test_metrics.true_negatives,
            "false_positives": test_metrics.false_positives,
            "false_negatives": test_metrics.false_negatives,
            "true_positives": test_metrics.true_positives,
        }
        mlflow.log_dict(cm_artifact, artifact_file="confusion_matrix.json")

        # 5. Audit tags
        default_tags = {
            "author": "Roland Tanácsi",
            "neptun": "HCY41E",
            "project": "mlsecops-coffee-quality",
            "framework": "scikit-learn",
        }
        if tags:
            default_tags.update(tags)
        mlflow.set_tags(default_tags)

        # 6. Serialized model pipeline
        mlflow_sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format=mlflow_sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            registered_model_name=registered_model_name,
        )

        return run_id


def promote_model_alias(
    model_name: str,
    version: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> None:
    """Assign an alias (e.g. 'champion' for production) to a specific registered model version.

    Args:
        model_name: Name of the registered model.
        version: Version string to alias.
        alias: Alias tag to assign (default 'champion').
        tracking_uri: Optional tracking URI override.
    """
    client = get_client(tracking_uri)
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)


def load_registered_model(
    model_name: str,
    alias: str = "champion",
    tracking_uri: str | None = None,
) -> Pipeline:
    """Load a production-ready model pipeline from the Model Registry using its alias.

    Args:
        model_name: Name of the registered model.
        alias: Model alias (default 'champion').
        tracking_uri: Optional tracking URI override.

    Returns:
        The loaded and deserialized scikit-learn Pipeline.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow_sklearn.load_model(model_uri)
    return cast(Pipeline, model)


def load_model_by_run_id(
    run_id: str,
    artifact_path: str = "model",
    tracking_uri: str | None = None,
) -> Pipeline:
    """Load a model pipeline directly from an MLflow run ID and artifact path.

    Args:
        run_id: The MLflow run ID string.
        artifact_path: The artifact subpath within the run (default 'model').
        tracking_uri: Optional tracking URI override.

    Returns:
        The loaded and deserialized scikit-learn Pipeline.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow_sklearn.load_model(model_uri)
    return cast(Pipeline, model)
