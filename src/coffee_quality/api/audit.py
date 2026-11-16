"""Structured audit logging for inference tracking, compliance, and drift detection.

Captures all prediction requests, input features, model outputs, latency metrics,
and operational metadata in an append-only, thread-safe JSONL audit log.
Provides redaction of sensitive attributes and convenient tabular extraction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any

from fastapi import Depends
import pandas as pd

from coffee_quality.api.dependencies import get_settings_dependency
from coffee_quality.config import Settings

logger = logging.getLogger(__name__)

SENSITIVE_PATTERNS = (
    "password",
    "secret",
    "token",
    "api_key",
    "authorization",
    "auth",
    "credential",
    "cookie",
    "private_key",
)


def redact_sensitive_data(val: Any) -> Any:
    """Recursively scrub sensitive keys and credential patterns from structured data.

    Args:
        val: Input data structure (dictionary, list, or scalar).

    Returns:
        Sanitized copy of the data with sensitive values masked.
    """
    if isinstance(val, dict):
        sanitized: dict[str, Any] = {}
        for k, v in val.items():
            key_lower = str(k).lower()
            if any(pattern in key_lower for pattern in SENSITIVE_PATTERNS):
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = redact_sensitive_data(v)
        return sanitized
    if isinstance(val, list):
        return [redact_sensitive_data(item) for item in val]
    return val


@dataclass
class AuditRecord:
    """Structured immutable entry representing a model inference event."""

    request_id: str
    timestamp_utc: str
    model_version: str | None
    model_alias: str | None
    item_count: int
    inference_latency_ms: float
    inputs: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    client_ip: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert audit record to a serializable dictionary."""
        return asdict(self)


class AuditLogger:
    """Thread-safe append-only JSONL logger for inference events."""

    def __init__(self, log_path: Path | str, enabled: bool = True) -> None:
        """Initialize AuditLogger with destination file path and active status.

        Args:
            log_path: Path to the target JSONL file.
            enabled: Whether audit writes are active.
        """
        self.log_path = Path(log_path)
        self.enabled = enabled
        self._lock = threading.Lock()

    def log(
        self,
        *,
        request_id: str,
        model_version: str | None,
        model_alias: str | None,
        item_count: int,
        inference_latency_ms: float,
        inputs: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        client_ip: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp_utc: str | None = None,
    ) -> bool:
        """Create and append an audit record to the JSONL log file.

        Args:
            request_id: Unique correlation identifier for the request.
            model_version: Semantic version or run ID of the model.
            model_alias: Registry alias of the serving model.
            item_count: Total number of records in the batch.
            inference_latency_ms: Pipeline inference execution time.
            inputs: Raw feature dictionaries passed to the model.
            predictions: Output prediction dictionaries.
            client_ip: Optional client IP address.
            metadata: Optional additional request metadata.
            timestamp_utc: ISO 8601 UTC timestamp; defaults to now.

        Returns:
            True if successfully written, False if disabled or write failed.
        """
        if not self.enabled:
            return False

        record = AuditRecord(
            request_id=request_id,
            timestamp_utc=timestamp_utc or datetime.now(timezone.utc).isoformat(),
            model_version=model_version,
            model_alias=model_alias,
            item_count=item_count,
            inference_latency_ms=inference_latency_ms,
            inputs=inputs,
            predictions=predictions,
            client_ip=client_ip,
            metadata=metadata,
        )
        return self.write_record(record)

    def write_record(self, record: AuditRecord) -> bool:
        """Safely write a single AuditRecord to disk with file locking and error handling.

        Args:
            record: Structured AuditRecord instance.

        Returns:
            True if written successfully, False otherwise.
        """
        if not self.enabled:
            return False

        try:
            record_dict = redact_sensitive_data(record.to_dict())
            json_line = json.dumps(record_dict, default=str) + "\n"

            with self._lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json_line)
                    f.flush()
            return True
        except Exception as exc:
            logger.error(
                "Failed to write audit record %s to %s: %s",
                record.request_id,
                self.log_path,
                exc,
            )
            return False

    def read_records(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Read and parse records from the JSONL log file.

        Args:
            limit: Optional maximum number of recent records to return.

        Returns:
            List of parsed audit record dictionaries in chronological order.
        """
        if not self.log_path.is_file():
            return []

        records: list[dict[str, Any]] = []
        with self._lock:
            try:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, start=1):
                        stripped = line.strip()
                        if not stripped:
                            continue
                        try:
                            item = json.loads(stripped)
                            if isinstance(item, dict):
                                records.append(item)
                        except json.JSONDecodeError as err:
                            logger.warning(
                                "Corrupted JSON on line %d of %s: %s",
                                line_num,
                                self.log_path,
                                err,
                            )
            except Exception as exc:
                logger.error("Failed reading audit log %s: %s", self.log_path, exc)
                return []

        if limit is not None and limit > 0:
            return records[-limit:]
        return records

    def to_dataframe(self, limit: int | None = None) -> pd.DataFrame:
        """Flatten audit logs into a tabular pandas DataFrame for drift monitoring.

        Extracts input feature columns and merges model predictions, request ID,
        and prediction timestamp.

        Args:
            limit: Maximum number of audit records to process.

        Returns:
            DataFrame containing flattened input features and prediction targets.
        """
        records = self.read_records(limit=limit)
        rows: list[dict[str, Any]] = []

        for rec in records:
            req_id = rec.get("request_id")
            ts = rec.get("timestamp_utc")
            inputs = rec.get("inputs", [])
            preds = rec.get("predictions", [])

            for idx, inp in enumerate(inputs):
                row = dict(inp)
                row["_audit_request_id"] = req_id
                row["_audit_timestamp_utc"] = ts

                if idx < len(preds):
                    pred = preds[idx]
                    row["Is_Specialty"] = pred.get("is_specialty")
                    row["predicted_probability"] = pred.get("probability")
                    row["predicted_confidence"] = pred.get("confidence")
                    row["predicted_label"] = pred.get("label")

                rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


def get_audit_logger(
    settings: Settings = Depends(get_settings_dependency),
) -> AuditLogger:
    """Dependency injecting a configured AuditLogger instance.

    Args:
        settings: Active application configuration.

    Returns:
        Configured AuditLogger instance.
    """
    return AuditLogger(
        log_path=settings.audit_log_path,
        enabled=settings.enable_audit_logging,
    )
