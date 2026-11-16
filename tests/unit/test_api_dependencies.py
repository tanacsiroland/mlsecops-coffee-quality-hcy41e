"""Unit tests for FastAPI dependency injection and safe model loading."""

from pathlib import Path
from unittest.mock import MagicMock

from fastapi import HTTPException
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from coffee_quality.api.dependencies import (
    ModelContainer,
    ModelMetadata,
    get_model_metadata,
    get_model_pipeline,
    get_settings_dependency,
    global_model_container,
    load_production_model,
)
from coffee_quality.config import Settings
from coffee_quality.models.baseline import build_dummy_pipeline
from coffee_quality.models.registry import init_mlflow, log_experiment_run, promote_model_alias
from coffee_quality.security.integrity import safe_save_model


@pytest.fixture(autouse=True)
def reset_global_container():
    """Ensure global model container is clean before and after each test."""
    global_model_container.clear()
    yield
    global_model_container.clear()


@pytest.fixture
def dummy_pipeline() -> Pipeline:
    """Create a minimal fitted Pipeline for testing."""
    pipeline = Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    pipeline.fit([[1, 2], [3, 4]], [1, 1])
    return pipeline


def test_model_container_lifecycle(dummy_pipeline: Pipeline):
    """Verify in-memory model cache operations."""
    container = ModelContainer()
    assert container.get() is None

    meta = container.set(dummy_pipeline, version="v1.0", alias="champion")
    assert isinstance(meta, ModelMetadata)
    assert meta.version == "v1.0"
    assert meta.alias == "champion"
    assert container.get() == meta

    container.clear()
    assert container.get() is None


def test_get_settings_dependency():
    """Verify settings dependency retrieval from app state and fallback."""
    mock_request = MagicMock()
    custom_settings = Settings(api_title="Custom Injection Test")
    mock_request.app.state.settings = custom_settings

    assert get_settings_dependency(mock_request) == custom_settings

    mock_request.app.state.settings = None
    default_settings = get_settings_dependency(mock_request)
    assert isinstance(default_settings, Settings)


def test_load_production_model_from_signed_local_artifact(
    tmp_path: Path,
    dummy_pipeline: Pipeline,
):
    """Verify loading local model artifact verified with cryptographic manifest."""
    model_path = tmp_path / "champion.bin"
    manifest_path = tmp_path / "champion.manifest.json"

    safe_save_model(
        model=dummy_pipeline,
        artifact_path=model_path,
        manifest_path=manifest_path,
        metadata={"version": "v1.2.3"},
    )

    settings = Settings(
        model_artifacts_dir=tmp_path,
        enable_artifact_integrity_check=True,
    )

    meta = load_production_model(settings)
    assert isinstance(meta, ModelMetadata)
    assert meta.version == "v1.2.3"
    assert isinstance(meta.model, Pipeline)


def test_load_production_model_local_integrity_tampering_raises_500(
    tmp_path: Path,
    dummy_pipeline: Pipeline,
):
    """Verify that a tampered local model artifact raises HTTP 500 before loading."""
    model_path = tmp_path / "champion.bin"
    manifest_path = tmp_path / "champion.manifest.json"

    safe_save_model(dummy_pipeline, model_path, manifest_path)

    # Tamper with the artifact
    model_path.write_bytes(b"corrupted_tampered_payload")

    settings = Settings(
        model_artifacts_dir=tmp_path,
        enable_artifact_integrity_check=True,
    )

    with pytest.raises(HTTPException) as exc_info:
        load_production_model(settings)

    assert exc_info.value.status_code == 500
    assert "integrity check failed" in str(exc_info.value.detail)


def test_load_production_model_local_without_integrity_check(
    tmp_path: Path,
    dummy_pipeline: Pipeline,
):
    """Verify loading local model when integrity check is disabled."""
    model_path = tmp_path / "champion.bin"
    manifest_path = tmp_path / "champion.manifest.json"

    safe_save_model(dummy_pipeline, model_path, manifest_path)

    settings = Settings(
        model_artifacts_dir=tmp_path,
        enable_artifact_integrity_check=False,
    )

    meta = load_production_model(settings)
    assert isinstance(meta, ModelMetadata)


def test_load_production_model_from_mlflow_registry(tmp_path: Path):
    """Verify loading model from MLflow Model Registry via 'champion' alias."""
    db_file = tmp_path / "test_dep_mlflow.db"
    tracking_uri = f"sqlite:///{db_file}"
    init_mlflow(tracking_uri=tracking_uri, experiment_name="dep_test_exp")

    from coffee_quality.models.baseline import evaluate_cv
    from coffee_quality.models.evaluator import evaluate_predictions
    import pandas as pd

    pipeline = build_dummy_pipeline()
    X = pd.DataFrame({
        "Country.of.Origin": ["Ethiopia", "Colombia", "Brazil", "Guatemala"],
        "Variety": ["Typica", "Bourbon", "Caturra", "Typica"],
        "Processing.Method": ["Washed", "Natural", "Washed", "Washed"],
        "Color": ["Green", "Green", "Green", "Green"],
        "altitude_mean_meters": [1800.0, 1600.0, 1200.0, 1500.0],
        "Moisture": [0.11, 0.12, 0.10, 0.11],
        "Category.One.Defects": [0, 1, 0, 0],
        "Category.Two.Defects": [2, 3, 1, 2],
        "Quakers": [0, 0, 1, 0],
        "Grading.Date": ["2015-04-04", "2016-05-10", "2017-06-01", "2018-07-04"],
    })
    y = pd.Series([1, 0, 1, 0])
    pipeline.fit(X, y)

    cv_res = evaluate_cv(pipeline, X, y, n_splits=2)
    test_metrics = evaluate_predictions(y, [1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1])

    log_experiment_run(
        pipeline=pipeline,
        run_name="reg_run",
        params={"model": "dummy"},
        cv_results=cv_res,
        test_metrics=test_metrics,
        registered_model_name="coffee-quality-model",
    )
    promote_model_alias("coffee-quality-model", "1", alias="champion", tracking_uri=tracking_uri)

    settings = Settings(
        model_artifacts_dir=tmp_path / "non_existent_local_dir",
        mlflow_tracking_uri=tracking_uri,
    )

    meta = load_production_model(settings)
    assert isinstance(meta, ModelMetadata)
    assert meta.alias == "champion"
    assert isinstance(meta.model, Pipeline)


def test_load_production_model_missing_raises_503(tmp_path: Path):
    """Verify HTTP 503 is raised when neither local file nor MLflow model exists."""
    settings = Settings(
        model_artifacts_dir=tmp_path / "empty_dir",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'empty_mlflow.db'}",
    )

    with pytest.raises(HTTPException) as exc_info:
        load_production_model(settings)

    assert exc_info.value.status_code == 503
    assert "not currently available" in str(exc_info.value.detail)


def test_get_model_metadata_and_pipeline_resolution(dummy_pipeline: Pipeline):
    """Verify dependency caching behavior across app.state, global container, and loaders."""
    mock_request = MagicMock()
    mock_request.app.state.model = None
    mock_request.app.state.model_alias = None
    mock_request.app.state.model_version = None

    # 1. When global container has a cached model
    global_model_container.set(dummy_pipeline, version="v1.0", alias="champion")
    meta = get_model_metadata(mock_request)
    assert meta.version == "v1.0"
    assert mock_request.app.state.model == dummy_pipeline

    # 2. When request.app.state already has the model
    meta_from_state = get_model_metadata(mock_request)
    assert meta_from_state.model == dummy_pipeline

    # 3. get_model_pipeline injects the pipeline object
    pipeline_injected = get_model_pipeline(meta)
    assert pipeline_injected == dummy_pipeline

    # 4. When neither app.state nor global container has model, it triggers loader
    global_model_container.clear()
    mock_request.app.state.model = None
    mock_request.app.state.model_alias = None
    mock_request.app.state.model_version = None

    from unittest.mock import patch
    fake_meta = ModelMetadata(model=dummy_pipeline, version="mocked", alias="champion", loaded_at_utc="now")
    with patch("coffee_quality.api.dependencies.load_production_model", return_value=fake_meta):
        meta_loaded = get_model_metadata(mock_request)
        assert meta_loaded == fake_meta
        assert mock_request.app.state.model == dummy_pipeline
