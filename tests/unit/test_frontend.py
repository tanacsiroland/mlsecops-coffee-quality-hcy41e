"""Unit tests for the Streamlit frontend client and helper methods."""

from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from frontend.streamlit_app import (
    fetch_drift_html,
    fetch_drift_summary,
    fetch_service_health,
    get_api_url,
    predict_csv_batch,
    predict_single_lot,
    trigger_pipeline_retraining,
)


def test_get_api_url_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default fallback URL when no environment variables are set."""
    monkeypatch.delenv("COFFEE_API_URL", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    assert get_api_url() == "http://localhost:8000"


def test_get_api_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify URL retrieval from environment variables with trailing slash stripped."""
    monkeypatch.setenv("COFFEE_API_URL", "http://production-api:8000/")
    assert get_api_url() == "http://production-api:8000"


def test_fetch_service_health_success() -> None:
    """Verify successful parsing of health endpoint response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "healthy",
        "model_loaded": True,
        "app_env": "production",
        "model_alias": "champion",
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        health = fetch_service_health("http://mock-api")
        assert health is not None
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
        assert health["model_alias"] == "champion"


def test_fetch_service_health_failure_returns_none() -> None:
    """Verify graceful handling when serving service is unreachable."""
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        health = fetch_service_health("http://bad-api")
        assert health is None


def test_predict_single_lot_success() -> None:
    """Verify prediction submission for an individual sample."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "predictions": [
            {
                "is_specialty": 1,
                "probability": 0.88,
                "confidence": 0.88,
                "label": "Specialty",
            }
        ]
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        res, err = predict_single_lot({"country_of_origin": "Ethiopia"}, api_url="http://mock-api")
        assert err is None
        assert res is not None
        assert res["is_specialty"] == 1
        assert res["label"] == "Specialty"


def test_predict_single_lot_error() -> None:
    """Verify error reporting on serving failure."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Model not available"

    with patch("httpx.Client.post", return_value=mock_resp):
        res, err = predict_single_lot({}, api_url="http://mock-api")
        assert res is None
        assert err is not None
        assert "503" in err


def test_predict_csv_batch_success() -> None:
    """Verify batch inference adds predictions columns to DataFrame."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "predictions": [
            {
                "is_specialty": 1,
                "probability": 0.92,
                "confidence": 0.92,
                "label": "Specialty",
            },
            {
                "is_specialty": 0,
                "probability": 0.15,
                "confidence": 0.85,
                "label": "Commercial",
            },
        ]
    }

    df = pd.DataFrame([
        {"Country.of.Origin": "Ethiopia", "altitude_mean_meters": 1800.0},
        {"Country.of.Origin": "Brazil", "altitude_mean_meters": 1100.0},
    ])

    with patch("httpx.Client.post", return_value=mock_resp):
        annotated, err = predict_csv_batch(df, api_url="http://mock-api")
        assert err is None
        assert annotated is not None
        assert len(annotated) == 2
        assert list(annotated["Predicted_Is_Specialty"]) == [1, 0]
        assert list(annotated["Classification"]) == ["Specialty", "Commercial"]


def test_predict_csv_batch_empty_dataframe() -> None:
    """Verify validation guard when empty DataFrame is supplied."""
    empty_df = pd.DataFrame()
    res, err = predict_csv_batch(empty_df, api_url="http://mock-api")
    assert res is None
    assert "no records" in str(err)


def test_fetch_drift_summary_success() -> None:
    """Verify drift summary retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "dataset_drift_detected": False,
        "share_of_drifted_features": 0.1,
    }

    with patch("httpx.Client.get", return_value=mock_resp):
        data, err = fetch_drift_summary(limit=100, api_url="http://mock-api")
        assert err is None
        assert data is not None
        assert data["dataset_drift_detected"] is False


def test_fetch_drift_html_success() -> None:
    """Verify drift HTML report retrieval."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body>Interactive Report</body></html>"

    with patch("httpx.Client.get", return_value=mock_resp):
        html, err = fetch_drift_html(limit=100, api_url="http://mock-api")
        assert err is None
        assert html is not None
        assert "Interactive Report" in html


def test_trigger_pipeline_retraining_success() -> None:
    """Verify triggering retraining pipeline."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "completed",
        "run_id": "run-abc-123",
        "metrics": {"macro_f1": 0.81, "roc_auc": 0.85},
    }

    with patch("httpx.Client.post", return_value=mock_resp):
        data, err = trigger_pipeline_retraining("random_forest", True, api_url="http://mock-api")
        assert err is None
        assert data is not None
        assert data["run_id"] == "run-abc-123"


def test_trigger_pipeline_retraining_409_conflict() -> None:
    """Verify handling when retraining lock is held by another process."""
    mock_resp = MagicMock()
    mock_resp.status_code = 409
    mock_resp.text = "Job running"

    with patch("httpx.Client.post", return_value=mock_resp):
        data, err = trigger_pipeline_retraining("random_forest", True, api_url="http://mock-api")
        assert data is None
        assert "already in progress" in str(err)
