"""Comprehensive end-to-end integration test suite.

Validates the complete serving and MLSecOps lifecycle:
1. Application startup & initial health checks.
2. Cold-boot inference rejection (HTTP 503).
3. Live pipeline retraining via POST /train and champion registry promotion.
4. Zero-downtime hot reloading of the in-memory champion model.
5. Live batch inference via POST /predict with calibrated probabilities.
6. Structured audit log recording and X-Request-ID correlation.
7. OWASP security middleware, anti-leakage guards, and DoS protections.
"""

import json
from pathlib import Path
import uuid

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from coffee_quality.api.app import create_app
from coffee_quality.api.dependencies import global_model_container
from coffee_quality.api.routes.train import training_lock
from coffee_quality.config import Settings


def _create_mock_dataset(path: Path, n_samples: int = 40) -> None:
    """Generate a synthetically valid CQI Arabica dataset for training pipelines."""
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


@pytest.fixture(autouse=True)
def clean_system_state():
    """Ensure in-memory container and training locks are reset between test runs."""
    global_model_container.clear()
    if training_lock.locked():
        training_lock.release()
    yield
    global_model_container.clear()
    if training_lock.locked():
        training_lock.release()


def test_end_to_end_api_lifecycle(tmp_path: Path) -> None:
    """Execute full real-world lifecycle from cold boot to retraining, inference, and audit trail."""
    raw_data_file = tmp_path / "data" / "dataset.csv"
    _create_mock_dataset(raw_data_file, n_samples=40)

    mlflow_db = tmp_path / "mlflow.db"
    artifacts_dir = tmp_path / "artifacts" / "models"
    audit_file = tmp_path / "audit" / "audit.jsonl"

    settings = Settings(
        app_env="testing",
        data_raw_path=raw_data_file,
        mlflow_tracking_uri=f"sqlite:///{mlflow_db}",
        model_artifacts_dir=artifacts_dir,
        audit_log_path=audit_file,
        enable_audit_logging=True,
    )

    app = create_app(settings=settings)
    client = TestClient(app)

    # 1. Root & Discovery Check
    root_res = client.get("/")
    assert root_res.status_code == 200
    root_data = root_res.json()
    assert root_data["docs_url"] == "/docs"
    assert root_data["health_url"] == "/health"

    # 2. Cold-Boot Health Check: No model should be loaded yet
    health_res1 = client.get("/health")
    assert health_res1.status_code == 200
    health_data1 = health_res1.json()
    assert health_data1["status"] == "healthy"
    assert health_data1["model_loaded"] is False
    assert health_data1["model_alias"] is None

    # 3. Cold-Boot Inference Rejection: Calling /predict without an available model returns 503
    cold_predict_payload = {
        "items": [
            {
                "country_of_origin": "Ethiopia",
                "altitude_mean_meters": 1850.0,
                "moisture": 0.11,
            }
        ]
    }
    cold_pred_res = client.post("/predict", json=cold_predict_payload)
    assert cold_pred_res.status_code == 503
    assert "not currently available" in cold_pred_res.json()["detail"]

    # 4. Trigger Retraining Pipeline: POST /train executes real MLflow tracking and champion promotion
    train_payload = {
        "model_type": "random_forest",
        "register_as_champion": True,
    }
    train_res = client.post("/train", json=train_payload)
    assert train_res.status_code == 200
    train_data = train_res.json()

    assert train_data["status"] == "completed"
    assert train_data["model_type"] == "random_forest"
    assert train_data["promoted_as_champion"] is True
    assert "run_id" in train_data
    assert "macro_f1" in train_data["metrics"]

    # 5. Hot-Reloaded Health Check: /health now immediately reflects loaded champion model
    health_res2 = client.get("/health")
    assert health_res2.status_code == 200
    health_data2 = health_res2.json()
    assert health_data2["model_loaded"] is True
    assert health_data2["model_alias"] == "champion"

    # 6. Live Inference: POST /predict with batch of coffee lot samples
    batch_payload = {
        "items": [
            {
                "Country.of.Origin": "Ethiopia",
                "Variety": "Typica",
                "Processing.Method": "Washed",
                "Color": "Green",
                "altitude_mean_meters": 1950.0,
                "Moisture": 0.10,
                "Category.One.Defects": 0,
                "Category.Two.Defects": 1,
                "Quakers": 0,
                "Grading.Date": "2021-04-12",
            },
            {
                "country_of_origin": "Brazil",
                "variety": "Bourbon",
                "processing_method": "Natural",
                "color": "Green",
                "altitude_mean_meters": 1100.0,
                "moisture": 0.12,
                "category_one_defects": 2,
                "category_two_defects": 7,
                "quakers": 2,
                "grading_date": "2021-05-18",
            },
        ]
    }
    pred_res = client.post("/predict", json=batch_payload)
    assert pred_res.status_code == 200
    pred_data = pred_res.json()

    # Verify tracing correlation ID
    request_id = pred_res.headers.get("X-Request-ID")
    assert request_id is not None
    uuid.UUID(request_id)

    # Verify OWASP security headers
    assert pred_res.headers["X-Content-Type-Options"] == "nosniff"
    assert pred_res.headers["X-Frame-Options"] == "DENY"
    assert "Strict-Transport-Security" in pred_res.headers

    # Verify prediction outputs
    assert pred_data["count"] == 2
    assert len(pred_data["predictions"]) == 2
    for item in pred_data["predictions"]:
        assert item["is_specialty"] in (0, 1)
        assert 0.0 <= item["probability"] <= 1.0
        assert 0.5 <= item["confidence"] <= 1.0
        assert item["label"] in ("Specialty", "Commercial")

    # 7. Audit Log Trail Verification: Inspect generated JSONL file
    assert audit_file.is_file()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    audit_entry = json.loads(lines[0])
    assert audit_entry["request_id"] == request_id
    assert audit_entry["model_alias"] == "champion"
    assert audit_entry["item_count"] == 2
    assert len(audit_entry["inputs"]) == 2
    assert len(audit_entry["predictions"]) == 2

    # Verify recorded features and column names match baseline format
    first_input = audit_entry["inputs"][0]
    assert first_input["Country.of.Origin"] == "Ethiopia"
    assert first_input["altitude_mean_meters"] == 1950.0
    assert audit_entry["predictions"][0]["is_specialty"] == pred_data["predictions"][0]["is_specialty"]


def test_api_security_defenses_and_validation(tmp_path: Path) -> None:
    """Verify security controls: anti-leakage, range boundaries, DoS limits, and concurrency locks."""
    raw_data_file = tmp_path / "data" / "dataset.csv"
    _create_mock_dataset(raw_data_file, n_samples=40)

    settings = Settings(
        data_raw_path=raw_data_file,
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'sec_mlflow.db'}",
        audit_log_path=tmp_path / "audit.jsonl",
    )
    app = create_app(settings=settings)
    client = TestClient(app)

    # Train model first so /predict is active
    train_res = client.post("/train", json={"model_type": "random_forest", "register_as_champion": True})
    assert train_res.status_code == 200

    # 1. Anti-Leakage Gate: Target injection
    target_leak = {"items": [{"Is_Specialty": 1}]}
    res1 = client.post("/predict", json=target_leak)
    assert res1.status_code == 422

    # 2. Anti-Leakage Gate: Post-cupping score injection
    score_leak = {"items": [{"Total.Cup.Points": 86.5}]}
    res2 = client.post("/predict", json=score_leak)
    assert res2.status_code == 422

    # 3. Physical Boundary Enforcement: Altitude > 4000m
    high_alt = {"items": [{"altitude_mean_meters": 4500.0}]}
    res3 = client.post("/predict", json=high_alt)
    assert res3.status_code == 422

    # 4. Physical Boundary Enforcement: Moisture > 1.0
    high_moist = {"items": [{"moisture": 1.2}]}
    res4 = client.post("/predict", json=high_moist)
    assert res4.status_code == 422

    # 5. DoS Gate: Empty batch
    empty_batch = {"items": []}
    res5 = client.post("/predict", json=empty_batch)
    assert res5.status_code == 422

    # 6. DoS Gate: Oversized batch (> 500 items)
    oversized = {"items": [{"country_of_origin": "Kenya"} for _ in range(501)]}
    res6 = client.post("/predict", json=oversized)
    assert res6.status_code == 422

    # 7. Concurrency Lock: Simultaneous retraining requests rejected with 409
    assert training_lock.acquire(blocking=False)
    try:
        concurrent_train = client.post("/train", json={"model_type": "random_forest"})
        assert concurrent_train.status_code == 409
        assert "currently running" in concurrent_train.json()["detail"]
    finally:
        training_lock.release()

    # 8. OWASP Security Headers across multiple routes
    for path in ["/", "/health"]:
        resp = client.get(path)
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "DENY"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "Strict-Transport-Security" in resp.headers
