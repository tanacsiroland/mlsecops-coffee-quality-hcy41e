"""Unit tests for the application configuration module."""

from pathlib import Path
from typing import Generator

from pydantic import ValidationError
import pytest

from coffee_quality.config import Settings, clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def _reset_config_cache() -> Generator[None, None, None]:
    """Ensure settings cache is cleared before and after each test."""
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_default_settings() -> None:
    """Verify that default settings match baseline specifications."""
    settings = Settings()

    assert settings.app_env == "development"
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.api_title == "Coffee Quality Classification Service"
    assert settings.api_version == "0.1.0"
    assert settings.data_raw_path == Path("data/raw/dataset.csv")
    assert settings.data_processed_dir == Path("data/processed")
    assert settings.audit_log_path == Path("data/audit/audit.jsonl")
    assert settings.enable_audit_logging is True
    assert settings.model_artifacts_dir == Path("artifacts/models")
    assert settings.mlflow_tracking_uri == "sqlite:///mlflow.db"
    assert settings.mlflow_experiment_name == "coffee-quality-evaluation"
    assert settings.random_state == 42
    assert settings.specialty_score_threshold == 80.0
    assert settings.enable_artifact_integrity_check is True
    assert settings.max_request_payload_bytes == 1_048_576


def test_environment_variable_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that environment variables cleanly override default values."""
    monkeypatch.setenv("COFFEE_APP_ENV", "production")
    monkeypatch.setenv("COFFEE_API_PORT", "9090")
    monkeypatch.setenv("COFFEE_DATA_RAW_PATH", "custom/data.csv")
    monkeypatch.setenv("COFFEE_ENABLE_ARTIFACT_INTEGRITY_CHECK", "false")
    monkeypatch.setenv("COFFEE_ENABLE_AUDIT_LOGGING", "false")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

    settings = Settings.from_env()

    assert settings.app_env == "production"
    assert settings.api_port == 9090
    assert settings.data_raw_path == Path("custom/data.csv")
    assert settings.enable_artifact_integrity_check is False
    assert settings.enable_audit_logging is False
    assert settings.mlflow_tracking_uri == "http://mlflow-server:5000"


def test_from_env_file(tmp_path: Path) -> None:
    """Verify loading settings from an explicit .env file."""
    env_file = tmp_path / "custom.env"
    env_file.write_text("COFFEE_API_PORT=8888\nCOFFEE_APP_ENV=testing\n", encoding="utf-8")

    settings = Settings.from_env(env_file=env_file)

    assert settings.api_port == 8888
    assert settings.app_env == "testing"


def test_from_default_dotenv_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fallback loading from a default .env file in the current working directory."""
    monkeypatch.chdir(tmp_path)
    default_env = tmp_path / ".env"
    default_env.write_text("COFFEE_API_PORT=8765\n", encoding="utf-8")

    settings = Settings.from_env()
    assert settings.api_port == 8765


def test_invalid_port_validation() -> None:
    """Verify that out-of-range port numbers raise a validation error."""
    with pytest.raises(ValidationError):
        Settings(api_port=99999)

    with pytest.raises(ValidationError):
        Settings(api_port=80)


def test_invalid_environment_validation() -> None:
    """Verify that unsupported environments raise a validation error."""
    with pytest.raises(ValidationError):
        Settings(app_env="invalid_environment")  # type: ignore[arg-type]


def test_invalid_score_threshold_validation() -> None:
    """Verify that out-of-range score thresholds raise a validation error."""
    with pytest.raises(ValidationError):
        Settings(specialty_score_threshold=150.0)

    with pytest.raises(ValidationError):
        Settings(specialty_score_threshold=-5.0)


def test_get_settings_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that get_settings returns a cached instance until invalidated."""
    first = get_settings()
    second = get_settings()
    assert first is second

    monkeypatch.setenv("COFFEE_API_PORT", "8888")
    still_cached = get_settings()
    assert still_cached.api_port == 8000

    clear_settings_cache()
    fresh = get_settings()
    assert fresh.api_port == 8888
    assert fresh is not first
