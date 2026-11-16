"""Shared test fixtures."""

import pytest


@pytest.fixture
def package_version() -> str:
    """Return the current package version fixture."""
    return "0.1.0"
