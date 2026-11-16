"""API integration tests for the /predict endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from coffee_quality.api.app import create_app
from coffee_quality.api.dependencies import global_model_container
from coffee_quality.features.preprocessor import build_feature_preprocessor
from coffee_quality.models.baseline import build_dummy_pipeline


class NoProbaClassifier(BaseEstimator, ClassifierMixin):
    """Classifier lacking predict_proba for fallback testing."""

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore[no-untyped-def]
        import numpy as np
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X: pd.DataFrame):  # type: ignore[no-untyped-def]
        return [1] * len(X)


class SingleColProbaClassifier(BaseEstimator, ClassifierMixin):
    """Classifier whose predict_proba returns a 1D column."""

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore[no-untyped-def]
        import numpy as np
        self.classes_ = np.array([1])
        return self

    def predict(self, X: pd.DataFrame):  # type: ignore[no-untyped-def]
        return [1] * len(X)

    def predict_proba(self, X: pd.DataFrame):  # type: ignore[no-untyped-def]
        import numpy as np
        return np.ones((len(X), 1))


@pytest.fixture(autouse=True)
def clean_container():
    """Ensure global model container is clean before and after tests."""
    global_model_container.clear()
    yield
    global_model_container.clear()


@pytest.fixture
def active_app() -> TestClient:
    """Create test client with pre-loaded baseline model pipeline."""
    app_instance = create_app()

    pipeline = build_dummy_pipeline()
    # Fit dummy on minimal mock features
    df = pd.DataFrame({
        "Country.of.Origin": ["Ethiopia", "Colombia"],
        "Variety": ["Typica", "Bourbon"],
        "Processing.Method": ["Washed", "Natural"],
        "Color": ["Green", "Green"],
        "altitude_mean_meters": [1800.0, 1600.0],
        "Moisture": [0.11, 0.12],
        "Category.One.Defects": [0, 1],
        "Category.Two.Defects": [2, 3],
        "Quakers": [0, 0],
        "Grading.Date": ["2015-04-04", "2016-05-10"],
    })
    y = pd.Series([1, 0])
    pipeline.fit(df, y)

    app_instance.state.model = pipeline
    app_instance.state.model_version = "v1.0-test"
    app_instance.state.model_alias = "champion"

    return TestClient(app_instance)


def test_predict_single_item_success(active_app: TestClient):
    """Verify single-item prediction payload returns valid prediction."""
    payload = {
        "items": [
            {
                "country_of_origin": "Ethiopia",
                "variety": "Typica",
                "processing_method": "Washed",
                "color": "Green",
                "altitude_mean_meters": 1950.0,
                "moisture": 0.11,
                "category_one_defects": 0,
                "category_two_defects": 1,
                "quakers": 0,
                "grading_date": "2022-04-10",
            }
        ]
    }
    response = active_app.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 1
    assert data["model_version"] == "v1.0-test"
    assert data["inference_latency_ms"] >= 0.0

    prediction = data["predictions"][0]
    assert prediction["is_specialty"] in (0, 1)
    assert 0.0 <= prediction["probability"] <= 1.0
    assert 0.5 <= prediction["confidence"] <= 1.0
    assert prediction["label"] in ("Specialty", "Commercial")


def test_predict_batch_items_success(active_app: TestClient):
    """Verify batch predictions with diverse attributes."""
    payload = {
        "items": [
            {
                "country_of_origin": "Colombia",
                "variety": "Caturra",
                "processing_method": "Washed",
                "color": "Green",
                "altitude_mean_meters": 1700.0,
                "moisture": 0.10,
                "category_one_defects": 0,
                "category_two_defects": 2,
                "quakers": 0,
                "grading_date": "2020-03-01",
            },
            {
                "Country.of.Origin": "Brazil",
                "Variety": "Bourbon",
                "Processing.Method": "Natural",
                "Color": "Bluish-Green",
                "altitude_mean_meters": 1100.0,
                "Moisture": 0.12,
                "Category.One.Defects": 2,
                "Category.Two.Defects": 8,
                "Quakers": 2,
                "Grading.Date": "2019-11-15",
            },
        ]
    }
    response = active_app.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_predict_fallback_for_classifier_without_proba():
    """Verify fallback when model does not implement predict_proba."""
    app_instance = create_app()
    pipeline = Pipeline([
        ("preprocessor", build_feature_preprocessor()),
        ("classifier", NoProbaClassifier()),
    ])
    df = pd.DataFrame({
        "Country.of.Origin": ["Ethiopia"],
        "Variety": ["Typica"],
        "Processing.Method": ["Washed"],
        "Color": ["Green"],
        "altitude_mean_meters": [1800.0],
        "Moisture": [0.11],
        "Category.One.Defects": [0],
        "Category.Two.Defects": [2],
        "Quakers": [0],
        "Grading.Date": ["2015-04-04"],
    })
    pipeline.fit(df, pd.Series([1]))

    app_instance.state.model = pipeline
    client = TestClient(app_instance)

    payload = {"items": [{"country_of_origin": "Ethiopia"}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predictions"][0]["is_specialty"] == 1


def test_predict_fallback_for_single_column_proba():
    """Verify handling when predict_proba returns a 1D column."""
    app_instance = create_app()
    pipeline = Pipeline([
        ("preprocessor", build_feature_preprocessor()),
        ("classifier", SingleColProbaClassifier()),
    ])
    df = pd.DataFrame({
        "Country.of.Origin": ["Ethiopia"],
        "Variety": ["Typica"],
        "Processing.Method": ["Washed"],
        "Color": ["Green"],
        "altitude_mean_meters": [1800.0],
        "Moisture": [0.11],
        "Category.One.Defects": [0],
        "Category.Two.Defects": [2],
        "Quakers": [0],
        "Grading.Date": ["2015-04-04"],
    })
    pipeline.fit(df, pd.Series([1]))

    app_instance.state.model = pipeline
    client = TestClient(app_instance)

    payload = {"items": [{"country_of_origin": "Ethiopia"}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["predictions"][0]["probability"] == 1.0


def test_predict_physical_bounds_validation_error(active_app: TestClient):
    """Verify HTTP 422 when altitude exceeds 4000 meters or moisture exceeds 1.0."""
    invalid_altitude = {
        "items": [{"altitude_mean_meters": 4500.0}]
    }
    res_alt = active_app.post("/predict", json=invalid_altitude)
    assert res_alt.status_code == 422

    invalid_moisture = {
        "items": [{"moisture": 1.5}]
    }
    res_moist = active_app.post("/predict", json=invalid_moisture)
    assert res_moist.status_code == 422


def test_predict_anti_leakage_rejection(active_app: TestClient):
    """Verify HTTP 422 when post-cupping scores or targets are injected."""
    leak_cup_points = {
        "items": [{"Total.Cup.Points": 85.5}]
    }
    res_points = active_app.post("/predict", json=leak_cup_points)
    assert res_points.status_code == 422

    leak_target = {
        "items": [{"Is_Specialty": 1}]
    }
    res_target = active_app.post("/predict", json=leak_target)
    assert res_target.status_code == 422


def test_predict_batch_bounds(active_app: TestClient):
    """Verify empty batch and oversized batch rejection with HTTP 422."""
    empty_batch = {"items": []}
    res_empty = active_app.post("/predict", json=empty_batch)
    assert res_empty.status_code == 422

    oversized_batch = {
        "items": [{"country_of_origin": "Ethiopia"} for _ in range(501)]
    }
    res_over = active_app.post("/predict", json=oversized_batch)
    assert res_over.status_code == 422


def test_predict_model_unavailable_raises_503(tmp_path: Path):
    """Verify HTTP 503 is returned when no model is available in the system."""
    from coffee_quality.config import Settings
    isolated_settings = Settings(
        model_artifacts_dir=tmp_path / "empty_artifacts",
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'empty.db'}",
    )
    fresh_app = create_app(settings=isolated_settings)
    client = TestClient(fresh_app)

    payload = {"items": [{"country_of_origin": "Kenya"}]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
    assert "not currently available" in response.json()["detail"]
