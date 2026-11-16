"""Data module for coffee quality classification."""

from coffee_quality.data.loader import (
    REQUIRED_COLUMNS,
    compute_file_sha256,
    load_raw_data,
    verify_dataset_integrity,
)
from coffee_quality.data.schema import CoffeeFeatures, CoffeeRecord

__all__ = [
    "CoffeeFeatures",
    "CoffeeRecord",
    "REQUIRED_COLUMNS",
    "compute_file_sha256",
    "load_raw_data",
    "verify_dataset_integrity",
]
