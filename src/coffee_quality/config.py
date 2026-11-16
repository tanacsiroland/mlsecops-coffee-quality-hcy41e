"""Application configuration and typed settings module.

Loads environment variables, validates constraints, and provides a centralized
source of truth for service parameters, storage paths, and security policies.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Central typed configuration for the coffee quality service."""

    app_env: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Application lifecycle environment",
    )
    api_host: str = Field(
        default="0.0.0.0",
        description="Network interface for the FastAPI serving daemon",
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="TCP listening port for the API",
    )
    api_title: str = Field(
        default="Coffee Quality Classification Service",
        description="Public API title displayed in Swagger documentation",
    )
    api_version: str = Field(
        default="0.1.0",
        description="Service semantic version",
    )

    data_raw_path: Path = Field(
        default=Path("data/raw/dataset.csv"),
        description="Filesystem path to the raw input dataset",
    )
    data_processed_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory storing sanitized datasets and reference baseline",
    )
    audit_log_path: Path = Field(
        default=Path("data/audit/audit.jsonl"),
        description="Structured JSONL audit log capturing inference inputs and predictions",
    )
    model_artifacts_dir: Path = Field(
        default=Path("artifacts/models"),
        description="Local directory storing exported model weights and manifests",
    )

    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking backend URI (SQLite locally, HTTP service in Compose)",
    )
    mlflow_experiment_name: str = Field(
        default="coffee-quality-evaluation",
        description="MLflow experiment bucket for tracking training runs",
    )

    random_state: int = Field(
        default=42,
        description="Deterministic seed for train-test splitting and model training",
    )
    specialty_score_threshold: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="SCA cup points boundary demarcating specialty vs commercial grade",
    )
    enable_artifact_integrity_check: bool = Field(
        default=True,
        description="Enforces cryptographic SHA-256 verification before deserializing model artifacts",
    )
    max_request_payload_bytes: int = Field(
        default=1_048_576,
        gt=0,
        description="Maximum accepted request body size in bytes to prevent DoS attacks (default 1 MB)",
    )

    @classmethod
    def from_env(cls, env_file: Path | str | None = None) -> "Settings":
        """Build settings instance populated from environment variables with fallback defaults."""
        file_values: dict[str, str | None] = {}
        if env_file:
            file_values = dict(dotenv_values(env_file))
        elif Path(".env").is_file():
            file_values = dict(dotenv_values(".env"))

        overrides: dict[str, Any] = {}
        for field_name in cls.model_fields.keys():
            prefixed_key = f"COFFEE_{field_name.upper()}"
            bare_key = field_name.upper()

            value = os.getenv(prefixed_key)
            if value is None:
                value = os.getenv(bare_key)
            if value is None:
                value = file_values.get(prefixed_key)
            if value is None:
                value = file_values.get(bare_key)

            if value is not None:
                overrides[field_name] = value

        return cls.model_validate(overrides)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton loaded from environment."""
    return Settings.from_env()


def clear_settings_cache() -> None:
    """Invalidate the LRU cache of get_settings, primarily used during test suites."""
    get_settings.cache_clear()
