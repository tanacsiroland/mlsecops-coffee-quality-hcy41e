"""Skeleton test verifying package import and version definition."""

from coffee_quality import __version__


def test_package_version_defined(package_version: str) -> None:
    """Verify package version matches expected metadata."""
    assert __version__ == package_version
