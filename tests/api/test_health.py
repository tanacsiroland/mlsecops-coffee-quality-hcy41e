"""Unit tests for the FastAPI application skeleton, security headers, and health endpoint."""

from fastapi.testclient import TestClient
import pytest

from coffee_quality.api.app import app, create_app
from coffee_quality.config import Settings


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI test client."""
    return TestClient(app)


def test_root_endpoint(client: TestClient):
    """Verify service discovery information on root route."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Coffee Quality Classification Service" in data["service"]
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"


def test_health_endpoint_default(client: TestClient):
    """Verify default health check status when no model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_env"] == "development"
    assert data["model_loaded"] is False
    assert data["model_alias"] is None
    assert "timestamp_utc" in data


def test_health_endpoint_with_active_model():
    """Verify health endpoint reports correct status when a model is active."""
    custom_app = create_app()
    custom_app.state.model = "dummy_fitted_model_object"
    custom_app.state.model_alias = "champion"

    custom_client = TestClient(custom_app)
    response = custom_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] is True
    assert data["model_alias"] == "champion"


def test_security_headers_present(client: TestClient):
    """Verify OWASP-compliant security headers are attached to every response."""
    response = client.get("/health")
    assert response.status_code == 200
    headers = response.headers

    assert headers.get("X-Content-Type-Options") == "nosniff"
    assert headers.get("X-Frame-Options") == "DENY"
    assert headers.get("X-XSS-Protection") == "1; mode=block"
    assert "Strict-Transport-Security" in headers


def test_cors_preflight(client: TestClient):
    """Verify CORS preflight headers are permitted."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("*", "http://localhost:8501")


def test_global_exception_handler_sanitizes_errors():
    """Verify unhandled exceptions return a sanitized JSON 500 error without leaking stack traces."""
    faulty_app = create_app()

    @faulty_app.get("/trigger-crash")
    def trigger_crash():  # type: ignore[no-untyped-def]
        raise RuntimeError("Sensitive database connection credentials or internal path details")

    faulty_client = TestClient(faulty_app, raise_server_exceptions=False)
    response = faulty_client.get("/trigger-crash")

    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Internal server error occurred."
    assert "credentials" not in response.text


def test_create_app_with_custom_settings():
    """Verify application factory respects custom settings overrides."""
    custom_settings = Settings(
        api_title="Custom Test API",
        api_version="1.2.3",
        app_env="production",
    )
    custom_app = create_app(settings=custom_settings)
    custom_client = TestClient(custom_app)

    root_res = custom_client.get("/")
    assert root_res.json()["service"] == "Custom Test API"
    assert root_res.json()["version"] == "1.2.3"

    health_res = custom_client.get("/health")
    assert health_res.json()["app_env"] == "production"


def test_lifespan_context():
    """Verify startup and shutdown lifecycle hooks execute cleanly."""
    lifecycle_app = create_app()
    with TestClient(lifecycle_app) as lc_client:
        res = lc_client.get("/health")
        assert res.status_code == 200
