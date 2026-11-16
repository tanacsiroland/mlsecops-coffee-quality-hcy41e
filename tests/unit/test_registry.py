"""Unit tests for MLflow experiment tracking and model registry integration."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from coffee_quality.config import Settings
from coffee_quality.features.preprocessor import extract_features_and_target
from coffee_quality.models.baseline import (
    CrossValidationResult,
    build_dummy_pipeline,
    evaluate_cv,
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


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """Create an isolated SQLite tracking URI for testing."""
    db_file = tmp_path / "test_mlflow.db"
    return f"sqlite:///{db_file}"


@pytest.fixture
def sample_fitted_pipeline_and_metrics() -> tuple[
    Pipeline,
    CrossValidationResult,
    EvaluationMetrics,
    pd.DataFrame,
]:
    """Provide a minimal fitted pipeline and evaluation metrics."""
    pipeline = build_dummy_pipeline()
    df = pd.DataFrame({
        "Country.of.Origin": ["Ethiopia", "Colombia", "Brazil", "Guatemala", "Mexico", "Costa Rica"],
        "Variety": ["Typica", "Bourbon", "Caturra", "Typica", "Catuai", "Bourbon"],
        "Processing.Method": ["Washed", "Natural", "Washed", "Washed", "Natural", "Washed"],
        "Color": ["Green", "Bluish-Green", "Green", "Green", "Green", "Bluish-Green"],
        "altitude_mean_meters": [1800.0, 1600.0, 1200.0, 1500.0, 1400.0, 1700.0],
        "Moisture": [0.11, 0.12, 0.10, 0.11, 0.10, 0.12],
        "Category.One.Defects": [0, 1, 0, 0, 1, 0],
        "Category.Two.Defects": [2, 3, 1, 2, 2, 1],
        "Quakers": [0, 0, 1, 0, 0, 0],
        "Grading.Date": [
            "April 4th, 2015",
            "May 10th, 2016",
            "June 1st, 2017",
            "July 4th, 2018",
            "August 15th, 2019",
            "September 20th, 2020",
        ],
        "Is_Specialty": [1, 1, 0, 1, 0, 1],
    })

    X, y = extract_features_and_target(df)
    cv_results = evaluate_cv(pipeline, X, y, n_splits=2, random_state=42)

    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    test_metrics = evaluate_predictions(y, preds, y_prob=[0.9, 0.9, 0.1, 0.9, 0.2, 0.8])

    return pipeline, cv_results, test_metrics, X


def test_init_mlflow_lifecycle(tracking_uri: str):
    """Test experiment initialization, ID generation, and idempotent retrieval."""
    exp_id_1 = init_mlflow(tracking_uri=tracking_uri, experiment_name="test_experiment")
    assert isinstance(exp_id_1, str)

    # Calling again with same name should retrieve existing ID
    exp_id_2 = init_mlflow(tracking_uri=tracking_uri, experiment_name="test_experiment")
    assert exp_id_1 == exp_id_2


def test_init_mlflow_default_settings(tmp_path: Path):
    """Test init_mlflow falling back to application settings."""
    db_file = tmp_path / "fallback.db"
    dummy_settings = Settings(
        mlflow_tracking_uri=f"sqlite:///{db_file}",
        mlflow_experiment_name="fallback-exp",
    )

    with patch("coffee_quality.models.registry.get_settings", return_value=dummy_settings):
        exp_id = init_mlflow()
        assert isinstance(exp_id, str)
        client = get_client()
        assert client is not None


def test_log_experiment_run_and_load_by_run_id(
    tracking_uri: str,
    sample_fitted_pipeline_and_metrics: tuple[
        Pipeline,
        CrossValidationResult,
        EvaluationMetrics,
        pd.DataFrame,
    ],
):
    """Test logging parameters, metrics, artifacts, and loading model back by run ID."""
    pipeline, cv_results, test_metrics, X = sample_fitted_pipeline_and_metrics
    init_mlflow(tracking_uri=tracking_uri, experiment_name="run_logging_test")

    params: dict[str, Any] = {
        "model_type": "DummyClassifier",
        "strategy": "most_frequent",
    }
    tags = {"experiment_phase": "unit_test"}

    run_id = log_experiment_run(
        pipeline=pipeline,
        run_name="test_dummy_run",
        params=params,
        cv_results=cv_results,
        test_metrics=test_metrics,
        tags=tags,
    )

    assert isinstance(run_id, str)

    client = get_client(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    # Assert logged params
    assert run.data.params["model_type"] == "DummyClassifier"
    assert run.data.params["strategy"] == "most_frequent"

    # Assert logged metrics
    assert "cv_macro_f1_mean" in run.data.metrics
    assert "test_accuracy" in run.data.metrics
    assert "test_roc_auc" in run.data.metrics

    # Assert logged audit tags
    assert run.data.tags["author"] == "Roland Tanácsi"
    assert run.data.tags["neptun"] == "HCY41E"
    assert run.data.tags["experiment_phase"] == "unit_test"

    # Test loading model by run_id (both with explicit URI and relying on active URI)
    loaded_pipeline = load_model_by_run_id(run_id=run_id, tracking_uri=tracking_uri)
    assert isinstance(loaded_pipeline, Pipeline)
    loaded_default = load_model_by_run_id(run_id=run_id)
    assert isinstance(loaded_default, Pipeline)
    preds = loaded_default.predict(X)
    assert len(preds) == len(X)


def test_log_experiment_run_without_roc_auc(
    tracking_uri: str,
    sample_fitted_pipeline_and_metrics: tuple[
        Pipeline,
        CrossValidationResult,
        EvaluationMetrics,
        pd.DataFrame,
    ],
):
    """Verify that omitting ROC-AUC (None) logs gracefully without errors."""
    pipeline, cv_results, _, _ = sample_fitted_pipeline_and_metrics
    init_mlflow(tracking_uri=tracking_uri, experiment_name="no_roc_auc_test")

    # Metrics with roc_auc = None
    metrics_without_auc = EvaluationMetrics(
        accuracy=1.0,
        roc_auc=None,
        macro_f1=1.0,
        precision_commercial=1.0,
        recall_commercial=1.0,
        f1_commercial=1.0,
        precision_specialty=1.0,
        recall_specialty=1.0,
        f1_specialty=1.0,
        true_negatives=1,
        false_positives=0,
        false_negatives=0,
        true_positives=3,
    )

    run_id = log_experiment_run(
        pipeline=pipeline,
        run_name="no_auc_run",
        params={"dummy": "val"},
        cv_results=cv_results,
        test_metrics=metrics_without_auc,
    )

    client = get_client(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    assert "test_roc_auc" not in run.data.metrics
    assert "test_accuracy" in run.data.metrics


def test_promote_model_alias_and_load_registered(
    tracking_uri: str,
    sample_fitted_pipeline_and_metrics: tuple[
        Pipeline,
        CrossValidationResult,
        EvaluationMetrics,
        pd.DataFrame,
    ],
):
    """Test model registration, alias promotion, and loading registered model."""
    pipeline, cv_results, test_metrics, X = sample_fitted_pipeline_and_metrics
    init_mlflow(tracking_uri=tracking_uri, experiment_name="registry_test")

    model_name = "coffee_quality_classifier"

    # Log and register model
    log_experiment_run(
        pipeline=pipeline,
        run_name="registry_run",
        params={"model": "dummy"},
        cv_results=cv_results,
        test_metrics=test_metrics,
        registered_model_name=model_name,
    )

    # Promote version 1 to champion alias
    promote_model_alias(
        model_name=model_name,
        version="1",
        alias="champion",
        tracking_uri=tracking_uri,
    )

    # Load model using champion alias (both with explicit URI and relying on active URI)
    registered_pipeline = load_registered_model(
        model_name=model_name,
        alias="champion",
        tracking_uri=tracking_uri,
    )
    assert isinstance(registered_pipeline, Pipeline)

    registered_default = load_registered_model(
        model_name=model_name,
        alias="champion",
    )
    assert isinstance(registered_default, Pipeline)
    preds = registered_default.predict(X)
    assert len(preds) == len(X)
