"""Raw dataset ingestion, column presence verification, and cryptographic hashing."""

import hashlib
from pathlib import Path

import pandas as pd

from coffee_quality.config import get_settings

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Country.of.Origin",
    "Variety",
    "Processing.Method",
    "Color",
    "altitude_mean_meters",
    "Moisture",
    "Category.One.Defects",
    "Category.Two.Defects",
    "Quakers",
    "Grading.Date",
    "Is_Specialty",
)


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 cryptographic checksum of a file in streaming 64KB chunks."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")

    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_integrity(path: Path, expected_sha256: str) -> bool:
    """Verify that a file on disk matches an expected SHA-256 checksum."""
    actual_sha256 = compute_file_sha256(path)
    return actual_sha256.lower() == expected_sha256.strip().lower()


def load_raw_data(path: Path | None = None) -> tuple[pd.DataFrame, str]:
    """Load, validate required column presence, and compute checksum for raw CSV dataset.

    Args:
        path: Optional explicit filesystem path. Defaults to configured data_raw_path.

    Returns:
        Tuple containing loaded DataFrame and its SHA-256 cryptographic hex digest.

    Raises:
        FileNotFoundError: If the target path does not exist.
        ValueError: If the file is empty or missing required schema columns.
    """
    target_path = path or get_settings().data_raw_path

    if not target_path.exists():
        raise FileNotFoundError(f"Raw dataset file does not exist: {target_path}")

    if not target_path.is_file():
        raise ValueError(f"Raw dataset path is not a regular file: {target_path}")

    if target_path.stat().st_size == 0:
        raise ValueError(f"Raw dataset file is empty (0 bytes): {target_path}")

    checksum = compute_file_sha256(target_path)
    df = pd.read_csv(target_path)

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset at {target_path} is missing required columns: {sorted(missing_columns)}"
        )

    return df, checksum
