"""Unit tests for the structured audit logging engine and drift preparation."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from unittest.mock import patch
import uuid

from fastapi.testclient import TestClient
import pandas as pd

from coffee_quality.api.app import create_app
from coffee_quality.api.audit import (
    AuditLogger,
    AuditRecord,
    get_audit_logger,
    redact_sensitive_data,
)
from coffee_quality.config import Settings
from coffee_quality.models.baseline import build_dummy_pipeline


def test_redact_sensitive_data_scrubs_secrets() -> None:
    """Verify that credentials and secret keys are masked across nested structures."""
    payload = {
        "user": "analyst",
        "password": "supersecretpassword",
        "api_key": "xyz123456",
        "nested": {
            "auth_token": "bearer-token-val",
            "normal_data": 42,
            "secret_setting": "classified",
        },
        "records": [
            {"cookie": "session=abc", "value": 10},
            {"private_key": "-----BEGIN...", "country": "Ethiopia"},
        ],
    }

    sanitized = redact_sensitive_data(payload)

    assert sanitized["user"] == "analyst"
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["auth_token"] == "[REDACTED]"
    assert sanitized["nested"]["secret_setting"] == "[REDACTED]"
    assert sanitized["nested"]["normal_data"] == 42
    assert sanitized["records"][0]["cookie"] == "[REDACTED]"
    assert sanitized["records"][0]["value"] == 10
    assert sanitized["records"][1]["private_key"] == "[REDACTED]"
    assert sanitized["records"][1]["country"] == "Ethiopia"


def test_redact_sensitive_data_leaves_scalars_untouched() -> None:
    """Verify non-collection scalars return unchanged."""
    assert redact_sensitive_data("plain text") == "plain text"
    assert redact_sensitive_data(1234) == 1234
    assert redact_sensitive_data(None) is None


def test_audit_logger_write_and_read(tmp_path: Path) -> None:
    """Verify that audit records are written to JSONL and cleanly deserialized."""
    log_file = tmp_path / "nested" / "audit.jsonl"
    logger = AuditLogger(log_path=log_file, enabled=True)

    success = logger.log(
        request_id="req-12345",
        model_version="v1.0",
        model_alias="champion",
        item_count=1,
        inference_latency_ms=12.5,
        inputs=[{"Country.of.Origin": "Kenya", "altitude_mean_meters": 1800.0}],
        predictions=[{"is_specialty": 1, "probability": 0.92, "confidence": 0.92, "label": "Specialty"}],
        client_ip="192.168.1.1",
        metadata={"source": "pytest"},
    )

    assert success is True
    assert log_file.is_file()

    records = logger.read_records()
    assert len(records) == 1
    record = records[0]

    assert record["request_id"] == "req-12345"
    assert record["model_version"] == "v1.0"
    assert record["model_alias"] == "champion"
    assert record["item_count"] == 1
    assert record["inference_latency_ms"] == 12.5
    assert record["client_ip"] == "192.168.1.1"
    assert record["metadata"] == {"source": "pytest"}
    assert record["inputs"][0]["Country.of.Origin"] == "Kenya"
    assert record["predictions"][0]["is_specialty"] == 1


def test_audit_logger_disabled_skips_writes(tmp_path: Path) -> None:
    """Verify that disabled audit logger produces no disk I/O."""
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_file, enabled=False)

    success = logger.log(
        request_id="req-disabled",
        model_version="v1.0",
        model_alias="champion",
        item_count=1,
        inference_latency_ms=5.0,
        inputs=[{"Country.of.Origin": "Brazil"}],
        predictions=[{"is_specialty": 0, "probability": 0.2, "confidence": 0.8, "label": "Commercial"}],
    )

    assert success is False
    assert not log_file.exists()
    assert logger.read_records() == []


def test_audit_logger_missing_file_returns_empty(tmp_path: Path) -> None:
    """Verify reading a non-existent file gracefully returns empty list."""
    logger = AuditLogger(log_path=tmp_path / "non_existent.jsonl")
    assert logger.read_records() == []
    assert logger.to_dataframe().empty


def test_audit_logger_corrupted_lines_resilience(tmp_path: Path) -> None:
    """Verify that corrupted JSON lines do not crash the reader and valid records are kept."""
    log_file = tmp_path / "audit.jsonl"
    log_file.write_text(
        '{"request_id": "valid-1", "inputs": [{"a": 1}], "predictions": [{"p": 1}]}\n'
        'CORRUPTED_JSON_LINE_THAT_IS_NOT_VALID_SYNTAX\n'
        '\n'
        '{"request_id": "valid-2", "inputs": [{"a": 2}], "predictions": [{"p": 2}]}\n',
        encoding="utf-8",
    )

    logger = AuditLogger(log_path=log_file)
    records = logger.read_records()

    assert len(records) == 2
    assert records[0]["request_id"] == "valid-1"
    assert records[1]["request_id"] == "valid-2"


def test_audit_logger_concurrent_writes(tmp_path: Path) -> None:
    """Verify thread-safe concurrent writes without line corruption."""
    log_file = tmp_path / "concurrent_audit.jsonl"
    logger = AuditLogger(log_path=log_file)

    def write_worker(worker_id: int) -> None:
        for i in range(15):
            logger.log(
                request_id=f"worker-{worker_id}-item-{i}",
                model_version="v1.0",
                model_alias="champion",
                item_count=1,
                inference_latency_ms=1.0,
                inputs=[{"worker": worker_id, "seq": i}],
                predictions=[{"is_specialty": 1, "probability": 0.8}],
            )

    thread_count = 8
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [executor.submit(write_worker, tid) for tid in range(thread_count)]
        for f in futures:
            f.result()

    records = logger.read_records()
    assert len(records) == thread_count * 15

    # Test limit parameter
    recent = logger.read_records(limit=10)
    assert len(recent) == 10


def test_audit_logger_to_dataframe(tmp_path: Path) -> None:
    """Verify transformation of audit records into a tabular DataFrame for drift monitoring."""
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_file)

    logger.log(
        request_id="req-df-1",
        model_version="v1.0",
        model_alias="champion",
        item_count=2,
        inference_latency_ms=10.0,
        inputs=[
            {"Country.of.Origin": "Guatemala", "altitude_mean_meters": 1500.0},
            {"Country.of.Origin": "Costa Rica", "altitude_mean_meters": 1600.0},
        ],
        predictions=[
            {"is_specialty": 1, "probability": 0.85, "confidence": 0.85, "label": "Specialty"},
            {"is_specialty": 0, "probability": 0.35, "confidence": 0.65, "label": "Commercial"},
        ],
    )

    df = logger.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Country.of.Origin" in df.columns
    assert "altitude_mean_meters" in df.columns
    assert "Is_Specialty" in df.columns
    assert "predicted_probability" in df.columns
    assert "_audit_request_id" in df.columns

    assert df.loc[0, "Country.of.Origin"] == "Guatemala"
    assert df.loc[0, "Is_Specialty"] == 1
    assert df.loc[1, "Country.of.Origin"] == "Costa Rica"
    assert df.loc[1, "Is_Specialty"] == 0


def test_audit_logger_write_failure_isolated(tmp_path: Path) -> None:
    """Verify that file write exceptions are logged and caught gracefully."""
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(log_path=log_file)

    record = AuditRecord(
        request_id="err-req",
        timestamp_utc="2026-11-16T00:00:00Z",
        model_version="v1",
        model_alias="champ",
        item_count=1,
        inference_latency_ms=1.0,
        inputs=[],
        predictions=[],
    )

    with patch("builtins.open", side_effect=OSError("Disk full")):
        result = logger.write_record(record)
        assert result is False


def test_audit_logger_read_failure_isolated(tmp_path: Path) -> None:
    """Verify that file read exceptions are logged and return empty list."""
    log_file = tmp_path / "audit.jsonl"
    log_file.write_text('{"a": 1}\n', encoding="utf-8")
    logger = AuditLogger(log_path=log_file)

    with patch("builtins.open", side_effect=OSError("Permission denied")):
        records = logger.read_records()
        assert records == []


def test_get_audit_logger_dependency() -> None:
    """Verify get_audit_logger instantiates with configured settings."""
    settings = Settings(
        audit_log_path=Path("custom/audit.jsonl"),
        enable_audit_logging=False,
    )
    logger = get_audit_logger(settings)
    assert logger.log_path == Path("custom/audit.jsonl")
    assert logger.enabled is False


def test_predict_endpoint_writes_audit_log(tmp_path: Path) -> None:
    """Verify that calling /predict writes a corresponding line into the audit log."""
    audit_file = tmp_path / "live_audit.jsonl"
    settings = Settings(
        audit_log_path=audit_file,
        enable_audit_logging=True,
    )

    app = create_app(settings=settings)
    pipeline = build_dummy_pipeline()
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
    app.state.model = pipeline
    app.state.model_version = "v1-audit-test"
    app.state.model_alias = "champion"

    client = TestClient(app)

    payload = {
        "items": [
            {
                "country_of_origin": "Ethiopia",
                "variety": "Typica",
                "processing_method": "Washed",
                "color": "Green",
                "altitude_mean_meters": 1800.0,
                "moisture": 0.11,
                "category_one_defects": 0,
                "category_two_defects": 2,
                "quakers": 0,
                "grading_date": "2015-04-04",
            }
        ]
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    # Validate UUID format
    uuid.UUID(request_id)

    assert audit_file.is_file()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["request_id"] == request_id
    assert entry["model_version"] == "v1-audit-test"
    assert entry["item_count"] == 1
    assert entry["inputs"][0]["Country.of.Origin"] == "Ethiopia"
    assert len(entry["predictions"]) == 1
