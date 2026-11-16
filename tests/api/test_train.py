"""Integration and unit tests for the /train model retraining endpoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from coffee_quality.api.app import create_app
from coffee_quality.api.dependencies import ModelMetadata
from coffee_quality.api.routes.train import training_lock
from coffee_quality.config import Settings
from coffee_quality.models.baseline import CrossValidationResult
from coffee_quality.models.evaluator import EvaluationMetrics


def _create_mock_metrics() -> EvaluationMetrics:
    """Helper creating standard test evaluation metrics."""
    return EvaluationMetrics(
        accuracy=0.85,
        roc_auc=0.84,
        macro_f1=0.78,
        precision_commercial=0.70,
        recall_commercial=0.65,
        f1_commercial=0.67,
        precision_specialty=0.88,
        recall_specialty=0.92,
        f1_specialty=0.90,
        true_negatives=20,
        false_positives=10,
        false_negatives=5,
        true_positives=65,
    )


def _create_mock_cv() -> CrossValidationResult:
    """Helper creating mock cross-validation results."""
    return CrossValidationResult(
        roc_auc_mean=0.82,
        roc_auc_std=0.03,
        macro_f1_mean=0.76,
        macro_f1_std=0.04,
        accuracy_mean=0.83,
        accuracy_std=0.02,
        recall_commercial_mean=0.65,
        recall_commercial_std=0.05,
        precision_commercial_mean=0.70,
        precision_commercial_std=0.06,
        fold_metrics=[],
    )


def test_train_endpoint_success_with_champion_promotion(tmp_path: Path) -> None:
    """Verify successful training trigger, metrics response, and champion hot reload."""
    settings = Settings(
        data_raw_path=tmp_path / "dataset.csv",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        audit_log_path=tmp_path / "audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    mock_pipeline = MagicMock(spec=Pipeline)
    mock_meta = ModelMetadata(
        model=mock_pipeline,
        version="v2.0-champion",
        alias="champion",
        loaded_at_utc="2026-11-16T00:00:00Z",
    )

    with (
        patch("coffee_quality.api.routes.train.run_training_pipeline") as mock_run,
        patch("coffee_quality.api.routes.train.load_production_model", return_value=mock_meta) as mock_load,
    ):
        mock_run.return_value = ("run-mock-12345", _create_mock_metrics(), _create_mock_cv())

        payload = {"model_type": "random_forest", "register_as_champion": True}
        response = client.post("/train", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "completed"
        assert data["run_id"] == "run-mock-12345"
        assert data["model_type"] == "random_forest"
        assert data["promoted_as_champion"] is True
        assert data["metrics"]["accuracy"] == 0.85
        assert data["metrics"]["macro_f1"] == 0.78
        assert data["metrics"]["roc_auc"] == 0.84

        mock_run.assert_called_once()
        mock_load.assert_called_once_with(settings)

        # Verify app.state model was updated
        assert app.state.model is mock_pipeline
        assert app.state.model_version == "v2.0-champion"


def test_train_endpoint_without_champion_promotion(tmp_path: Path) -> None:
    """Verify training without promotion does not trigger hot reloading."""
    settings = Settings(
        data_raw_path=tmp_path / "dataset.csv",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        audit_log_path=tmp_path / "audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    with (
        patch("coffee_quality.api.routes.train.run_training_pipeline") as mock_run,
        patch("coffee_quality.api.routes.train.load_production_model") as mock_load,
    ):
        mock_run.return_value = ("run-candidate-999", _create_mock_metrics(), _create_mock_cv())

        payload = {"model_type": "lightgbm", "register_as_champion": False}
        response = client.post("/train", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "run-candidate-999"
        assert data["promoted_as_champion"] is False
        mock_load.assert_not_called()


def test_train_endpoint_default_payload_when_empty_body(tmp_path: Path) -> None:
    """Verify POST /train with empty body defaults to random_forest champion promotion."""
    settings = Settings(
        data_raw_path=tmp_path / "dataset.csv",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        audit_log_path=tmp_path / "audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    with (
        patch("coffee_quality.api.routes.train.run_training_pipeline") as mock_run,
        patch("coffee_quality.api.routes.train.load_production_model") as mock_load,
    ):
        mock_run.return_value = ("run-default-001", _create_mock_metrics(), _create_mock_cv())
        mock_load.return_value = ModelMetadata(
            model=MagicMock(spec=Pipeline),
            version="v1.0",
            alias="champion",
            loaded_at_utc="2026-11-16T00:00:00Z",
        )

        response = client.post("/train")
        assert response.status_code == 200
        assert response.json()["model_type"] == "random_forest"
        assert response.json()["promoted_as_champion"] is True


def test_train_concurrency_lock_returns_409(tmp_path: Path) -> None:
    """Verify concurrent requests are rejected with HTTP 409 Conflict."""
    settings = Settings(
        data_raw_path=tmp_path / "dataset.csv",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        audit_log_path=tmp_path / "audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    # Artificially acquire the lock to simulate an ongoing training job
    assert training_lock.acquire(blocking=False)
    try:
        response = client.post("/train", json={"model_type": "random_forest"})
        assert response.status_code == 409
        assert "currently running" in response.json()["detail"]
    finally:
        training_lock.release()


def test_train_pipeline_failure_returns_500(tmp_path: Path) -> None:
    """Verify pipeline execution errors return HTTP 500 and release lock."""
    settings = Settings(
        data_raw_path=tmp_path / "dataset.csv",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        audit_log_path=tmp_path / "audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    with patch("coffee_quality.api.routes.train.run_training_pipeline", side_effect=RuntimeError("Dataset unreadable")):
        response = client.post("/train", json={"model_type": "random_forest"})
        assert response.status_code == 500
        assert "retraining execution failed" in response.json()["detail"]

    # Verify lock was released
    assert not training_lock.locked()


def test_train_hot_reload_failure_warning_handled(tmp_path: Path) -> None:
    """Verify hot-reload error does not fail the endpoint if training succeeded."""
    settings = Settings(
        data_raw_path=tmp_path / "dataset.csv",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'test.db'}",
        audit_log_path=tmp_path / "audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    with (
        patch("coffee_quality.api.routes.train.run_training_pipeline") as mock_run,
        patch("coffee_quality.api.routes.train.load_production_model", side_effect=Exception("Registry socket error")),
    ):
        mock_run.return_value = ("run-ok-123", _create_mock_metrics(), _create_mock_cv())

        response = client.post("/train", json={"register_as_champion": True})
        assert response.status_code == 200
        assert response.json()["status"] == "completed"


def test_train_invalid_model_type_returns_422(tmp_path: Path) -> None:
    """Verify unsupported model architecture raises validation error."""
    app = create_app()
    client = TestClient(app)

    response = client.post("/train", json={"model_type": "unsupported_architecture"})
    assert response.status_code == 422


def test_train_extra_fields_returns_422() -> None:
    """Verify extra attributes are rejected by strict schema validation."""
    app = create_app()
    client = TestClient(app)

    response = client.post("/train", json={"unrecognized_key": "val"})
    assert response.status_code == 422
