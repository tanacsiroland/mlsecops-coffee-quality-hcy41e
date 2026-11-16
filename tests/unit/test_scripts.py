"""Unit tests for the standalone REST API test client script."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx

from scripts.test_client import (
    DEFAULT_SAMPLES,
    download_drift_html,
    main,
    parse_args,
    run_drift_check,
    run_health_check,
    run_predict,
    run_retrain,
)


def _make_mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str | None = None,
) -> httpx.Response:
    request = httpx.Request("GET", "http://testserver")
    if json_data is not None:
        return httpx.Response(status_code=status_code, json=json_data, request=request)
    return httpx.Response(status_code=status_code, text=text or "", request=request)


def test_parse_args_defaults() -> None:
    """Verify default CLI arguments."""
    args = parse_args([])
    assert args.base_url == "http://localhost:8000"
    assert args.retrain is False
    assert args.model_type == "lightgbm"
    assert args.save_html is None
    assert args.audit_limit == 50


def test_parse_args_custom() -> None:
    """Verify custom CLI flags."""
    args = parse_args([
        "--base-url", "http://127.0.0.1:9000",
        "--retrain",
        "--model-type", "random_forest",
        "--save-html", "test.html",
        "--audit-limit", "100",
    ])
    assert args.base_url == "http://127.0.0.1:9000"
    assert args.retrain is True
    assert args.model_type == "random_forest"
    assert args.save_html == "test.html"
    assert args.audit_limit == 100


def test_run_health_check() -> None:
    """Verify health check querying."""
    mock_client = MagicMock(spec=httpx.Client)
    payload = {"status": "healthy", "model_loaded": True, "model_alias": "champion"}
    mock_client.get.return_value = _make_mock_response(200, json_data=payload)

    result = run_health_check(mock_client, "http://localhost:8000")
    assert result == payload
    mock_client.get.assert_called_once_with("http://localhost:8000/health", timeout=10.0)


def test_run_predict_default_and_custom() -> None:
    """Verify prediction querying with default and custom samples."""
    mock_client = MagicMock(spec=httpx.Client)
    payload = {
        "predictions": [
            {"classification_label": "Specialty", "specialty_probability": 0.95},
            {"classification_label": "Commercial", "specialty_probability": 0.20},
        ],
        "inference_latency_ms": 12.34,
        "model_version": "v1.0.0",
    }
    mock_client.post.return_value = _make_mock_response(200, json_data=payload)

    # Test with default samples
    res1 = run_predict(mock_client, "http://localhost:8000")
    assert res1 == payload
    mock_client.post.assert_called_with(
        "http://localhost:8000/predict",
        json={"items": DEFAULT_SAMPLES},
        timeout=15.0,
    )

    # Test with custom samples
    custom_samples = [{"country_of_origin": "Colombia"}]
    res2 = run_predict(mock_client, "http://localhost:8000", samples=custom_samples)
    assert res2 == payload
    mock_client.post.assert_called_with(
        "http://localhost:8000/predict",
        json={"items": custom_samples},
        timeout=15.0,
    )


def test_run_drift_check() -> None:
    """Verify drift check querying."""
    mock_client = MagicMock(spec=httpx.Client)
    payload = {
        "dataset_drift": False,
        "share_of_drifted_columns": 0.0,
        "reference_row_count": 1000,
        "current_row_count": 50,
    }
    mock_client.get.return_value = _make_mock_response(200, json_data=payload)

    result = run_drift_check(mock_client, "http://localhost:8000", limit=25)
    assert result == payload
    mock_client.get.assert_called_once_with(
        "http://localhost:8000/monitoring/drift",
        params={"limit": 25},
        timeout=30.0,
    )


def test_download_drift_html(tmp_path: Path) -> None:
    """Verify HTML drift report downloading and saving."""
    mock_client = MagicMock(spec=httpx.Client)
    html_content = "<html><body>Drift Report</body></html>"
    mock_client.get.return_value = _make_mock_response(200, text=html_content)

    out_file = tmp_path / "report.html"
    saved = download_drift_html(mock_client, "http://localhost:8000", output_path=out_file, limit=10)
    assert saved == out_file
    assert out_file.read_text(encoding="utf-8") == html_content
    mock_client.get.assert_called_once_with(
        "http://localhost:8000/monitoring/drift/html",
        params={"limit": 10},
        timeout=30.0,
    )


def test_run_retrain() -> None:
    """Verify retraining invocation."""
    mock_client = MagicMock(spec=httpx.Client)
    payload = {
        "run_id": "mock_run_123",
        "is_promoted": True,
        "metrics": {"f1_specialty": 0.92, "roc_auc": 0.94},
    }
    mock_client.post.return_value = _make_mock_response(200, json_data=payload)

    result = run_retrain(mock_client, "http://localhost:8000", model_type="lightgbm", promote=True)
    assert result == payload
    mock_client.post.assert_called_once_with(
        "http://localhost:8000/train",
        json={"model_type": "lightgbm", "promote_as_champion": True},
        timeout=120.0,
    )


def test_main_success_without_retrain(tmp_path: Path) -> None:
    """Verify successful execution flow without retraining."""
    html_target = str(tmp_path / "out.html")
    with patch("scripts.test_client.httpx.Client") as mock_client_cls:
        client_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client_instance

        client_instance.get.side_effect = [
            _make_mock_response(200, json_data={"status": "healthy"}),
            _make_mock_response(200, json_data={"dataset_drift": False}),
            _make_mock_response(200, text="<html>Report</html>"),
        ]
        client_instance.post.side_effect = [
            _make_mock_response(200, json_data={"predictions": []}),
        ]

        exit_code = main(["--save-html", html_target])
        assert exit_code == 0


def test_main_success_with_retrain() -> None:
    """Verify successful execution flow with retraining flag."""
    with patch("scripts.test_client.httpx.Client") as mock_client_cls:
        client_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client_instance

        client_instance.get.side_effect = [
            _make_mock_response(200, json_data={"status": "healthy"}),
            _make_mock_response(200, json_data={"dataset_drift": False}),
        ]
        client_instance.post.side_effect = [
            _make_mock_response(200, json_data={"predictions": []}),
            _make_mock_response(200, json_data={"run_id": "retrain_1"}),
        ]

        exit_code = main(["--retrain"])
        assert exit_code == 0


def test_main_connection_error() -> None:
    """Verify handling of connection errors."""
    with patch("scripts.test_client.httpx.Client") as mock_client_cls:
        client_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client_instance
        client_instance.get.side_effect = httpx.ConnectError("Connection refused")

        exit_code = main([])
        assert exit_code == 1


def test_main_http_status_error() -> None:
    """Verify handling of HTTP errors."""
    with patch("scripts.test_client.httpx.Client") as mock_client_cls:
        client_instance = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = client_instance
        req = httpx.Request("GET", "http://localhost:8000/health")
        resp = httpx.Response(500, request=req, text="Internal Server Error")
        client_instance.get.side_effect = httpx.HTTPStatusError("Server error", request=req, response=resp)

        exit_code = main([])
        assert exit_code == 2


def test_main_unexpected_error() -> None:
    """Verify handling of unexpected runtime errors."""
    with patch("scripts.test_client.httpx.Client") as mock_client_cls:
        mock_client_cls.side_effect = RuntimeError("Fatal hardware failure")

        exit_code = main([])
        assert exit_code == 3
