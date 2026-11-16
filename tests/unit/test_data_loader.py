"""Unit tests for dataset loading, column verification, and checksum hashing."""

from pathlib import Path

import pytest

from coffee_quality.data.loader import (
    compute_file_sha256,
    load_raw_data,
    verify_dataset_integrity,
)


def test_load_real_raw_dataset() -> None:
    """Verify loading the project's real raw dataset file."""
    df, checksum = load_raw_data()

    assert df.shape == (1310, 11)
    assert len(checksum) == 64
    assert isinstance(checksum, str)
    assert df["Is_Specialty"].nunique() == 2


def test_compute_file_sha256_missing_file(tmp_path: Path) -> None:
    """Verify that attempting to hash a non-existent file raises FileNotFoundError."""
    missing = tmp_path / "non_existent.csv"
    with pytest.raises(FileNotFoundError):
        compute_file_sha256(missing)


def test_load_raw_data_missing_file(tmp_path: Path) -> None:
    """Verify that loading from a non-existent path raises FileNotFoundError."""
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(missing)


def test_load_raw_data_directory_path(tmp_path: Path) -> None:
    """Verify that passing a directory path raises ValueError."""
    with pytest.raises(ValueError, match="not a regular file"):
        load_raw_data(tmp_path)


def test_load_raw_data_empty_file(tmp_path: Path) -> None:
    """Verify that an empty (0 bytes) CSV file raises ValueError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="is empty"):
        load_raw_data(empty_file)


def test_load_raw_data_missing_columns(tmp_path: Path) -> None:
    """Verify that a CSV missing required columns raises ValueError."""
    incomplete_csv = tmp_path / "incomplete.csv"
    incomplete_csv.write_text("Country.of.Origin,Variety\nEthiopia,Typica\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_raw_data(incomplete_csv)


def test_verify_dataset_integrity(tmp_path: Path) -> None:
    """Verify dataset integrity checking against expected SHA-256 digests."""
    sample_file = tmp_path / "sample.csv"
    sample_file.write_text("header1,header2\nval1,val2\n", encoding="utf-8")

    actual_checksum = compute_file_sha256(sample_file)

    assert verify_dataset_integrity(sample_file, actual_checksum) is True
    assert verify_dataset_integrity(sample_file, actual_checksum.upper()) is True
    assert verify_dataset_integrity(sample_file, f"  {actual_checksum}  ") is True
    assert verify_dataset_integrity(sample_file, "0" * 64) is False
