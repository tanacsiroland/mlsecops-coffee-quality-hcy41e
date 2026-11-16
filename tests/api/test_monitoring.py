"""Integration tests for the /monitoring API endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd

from coffee_quality.api.app import create_app
from coffee_quality.api.audit import AuditLogger
from coffee_quality.config import Settings


def _create_mock_dataset(path: Path, n_samples: int = 40) -> None:
    """Generate a synthetically valid CQI Arabica reference dataset."""
    countries = ["Ethiopia", "Colombia", "Brazil", "Guatemala", "Kenya"]
    varieties = ["Typica", "Bourbon", "Caturra", "SL28", "Geisha"]
    methods = ["Washed", "Natural", "Washed", "Semi-Washed", "Natural"]
    colors = ["Green", "Bluish-Green", "Green", "Green", "Bluish-Green"]

    data = {
        "Country.of.Origin": [countries[i % len(countries)] for i in range(n_samples)],
        "Variety": [varieties[i % len(varieties)] for i in range(n_samples)],
        "Processing.Method": [methods[i % len(methods)] for i in range(n_samples)],
        "Color": [colors[i % len(colors)] for i in range(n_samples)],
        "altitude_mean_meters": [1100.0 + (i * 25.0) for i in range(n_samples)],
        "Moisture": [0.08 + (i % 5) * 0.01 for i in range(n_samples)],
        "Category.One.Defects": [i % 3 for i in range(n_samples)],
        "Category.Two.Defects": [i % 6 for i in range(n_samples)],
        "Quakers": [0 if i % 4 != 0 else 1 for i in range(n_samples)],
        "Grading.Date": [f"2020-{(i % 8) + 1:02d}-15" for i in range(n_samples)],
        "Is_Specialty": [1 if i % 2 == 0 else 0 for i in range(n_samples)],
    }
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _populate_mock_audit_log(audit_path: Path, n_records: int = 15) -> None:
    """Populate an audit log with realistic inference events."""
    logger = AuditLogger(log_path=audit_path, enabled=True)
    for i in range(n_records):
        logger.log(
            request_id=f"audit-req-{i}",
            model_version="v1.0-test",
            model_alias="champion",
            item_count=1,
            inference_latency_ms=10.0,
            inputs=[
                {
                    "Country.of.Origin": "Ethiopia" if i % 2 == 0 else "Colombia",
                    "Variety": "Typica",
                    "Processing.Method": "Washed",
                    "Color": "Green",
                    "altitude_mean_meters": 1700.0 + (i * 10.0),
                    "Moisture": 0.10,
                    "Category.One.Defects": 0,
                    "Category.Two.Defects": 2,
                    "Quakers": 0,
                    "Grading.Date": "2021-04-12",
                }
            ],
            predictions=[
                {
                    "is_specialty": 1,
                    "probability": 0.88,
                    "confidence": 0.88,
                    "label": "Specialty",
                }
            ],
        )


def test_get_drift_summary_insufficient_samples(tmp_path: Path) -> None:
    """Verify drift summary reports insufficient_samples when log is empty."""
    ref_file = tmp_path / "reference.csv"
    _create_mock_dataset(ref_file, n_samples=30)

    settings = Settings(
        data_raw_path=ref_file,
        audit_log_path=tmp_path / "empty_audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/monitoring/drift")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "insufficient_samples"
    assert data["dataset_drift_detected"] is False
    assert data["current_sample_count"] == 0
    assert data["total_features_count"] == 0


def test_get_drift_html_report_insufficient_samples(tmp_path: Path) -> None:
    """Verify HTML report endpoint returns fallback message when samples are insufficient."""
    ref_file = tmp_path / "reference.csv"
    _create_mock_dataset(ref_file, n_samples=30)

    settings = Settings(
        data_raw_path=ref_file,
        audit_log_path=tmp_path / "empty_audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/monitoring/drift/html")
    assert response.status_code == 200
    assert "text/html" in response.headers["Content-Type"]
    assert "Insufficient Samples" in response.text


def test_get_drift_summary_with_audit_records(tmp_path: Path) -> None:
    """Verify successful statistical drift calculation over real audit log records."""
    ref_file = tmp_path / "reference.csv"
    _create_mock_dataset(ref_file, n_samples=40)

    audit_file = tmp_path / "live_audit.jsonl"
    _populate_mock_audit_log(audit_file, n_records=12)

    settings = Settings(
        data_raw_path=ref_file,
        audit_log_path=audit_file,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/monitoring/drift?limit=50")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["current_sample_count"] == 12
    assert data["reference_sample_count"] == 40
    assert data["total_features_count"] > 0
    assert "drift_by_feature" in data
    assert isinstance(data["dataset_drift_detected"], bool)


def test_get_drift_html_report_with_audit_records(tmp_path: Path) -> None:
    """Verify interactive HTML report returns rendered Evidently markup."""
    ref_file = tmp_path / "reference.csv"
    _create_mock_dataset(ref_file, n_samples=40)

    audit_file = tmp_path / "live_audit.jsonl"
    _populate_mock_audit_log(audit_file, n_records=10)

    settings = Settings(
        data_raw_path=ref_file,
        audit_log_path=audit_file,
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    response = client.get("/monitoring/drift/html?limit=50")
    assert response.status_code == 200
    assert "text/html" in response.headers["Content-Type"]
    assert len(response.text) > 1000


def test_monitoring_endpoints_parameter_validation(tmp_path: Path) -> None:
    """Verify query parameter bounds validation rejects out-of-range values."""
    app = create_app()
    client = TestClient(app)

    # limit below minimum 5
    res1 = client.get("/monitoring/drift?limit=2")
    assert res1.status_code == 422

    # limit exceeding maximum 10000
    res2 = client.get("/monitoring/drift?limit=20000")
    assert res2.status_code == 422

    # html endpoint validation
    res3 = client.get("/monitoring/drift/html?limit=0")
    assert res3.status_code == 422


def test_monitoring_owasp_headers(tmp_path: Path) -> None:
    """Verify OWASP security headers are present on monitoring endpoints."""
    ref_file = tmp_path / "reference.csv"
    _create_mock_dataset(ref_file, n_samples=30)

    settings = Settings(data_raw_path=ref_file, audit_log_path=tmp_path / "audit.jsonl")
    app = create_app(settings=settings)
    client = TestClient(app)

    for route in ["/monitoring/drift", "/monitoring/drift/html"]:
        res = client.get(route)
        assert res.headers.get("X-Content-Type-Options") == "nosniff"
        assert res.headers.get("X-Frame-Options") == "DENY"
        assert "Strict-Transport-Security" in res.headers
