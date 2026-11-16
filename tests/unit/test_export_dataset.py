"""Unit tests for the preprocessed dataset exporter script."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.export_processed_dataset import (
    DEFAULT_OUTPUT_PATH,
    export_processed_dataset,
    main,
    parse_args,
    print_summary_table,
)


def _make_dummy_raw_df() -> pd.DataFrame:
    """Generate minimal valid raw dataset for testing."""
    return pd.DataFrame({
        "Country.of.Origin": ["Ethiopia", "Guatemala"],
        "Variety": ["Typica", "Bourbon"],
        "Processing.Method": ["Washed", "Natural"],
        "Color": ["Green", "Bluish-Green"],
        "altitude_mean_meters": [1800.0, 1500.0],
        "Moisture": [0.11, 0.12],
        "Category.One.Defects": [0, 1],
        "Category.Two.Defects": [1, 2],
        "Quakers": [0, 1],
        "Grading.Date": ["2020-01-01", "2020-02-01"],
        "Is_Specialty": [1, 0],
    })


def test_parse_args_defaults() -> None:
    """Verify default CLI arguments."""
    args = parse_args([])
    assert args.raw_path is None
    assert args.output_path == DEFAULT_OUTPUT_PATH
    assert args.max_altitude == 4000.0


def test_parse_args_custom() -> None:
    """Verify custom CLI argument flags."""
    args = parse_args([
        "--raw-path", "data/raw/custom.csv",
        "--output-path", "data/processed/custom.csv",
        "--max-altitude", "3500.0",
    ])
    assert args.raw_path == Path("data/raw/custom.csv")
    assert args.output_path == Path("data/processed/custom.csv")
    assert args.max_altitude == 3500.0


def test_export_processed_dataset_success(tmp_path: Path) -> None:
    """Verify end-to-end export logic with mock data."""
    raw_csv = tmp_path / "raw.csv"
    out_csv = tmp_path / "processed" / "out.csv"

    dummy_df = _make_dummy_raw_df()
    dummy_df.to_csv(raw_csv, index=False)

    df, file_hash = export_processed_dataset(
        raw_path=raw_csv,
        output_path=out_csv,
        max_altitude=3000.0,
    )

    assert len(df) == 2
    assert out_csv.is_file()
    assert len(file_hash) == 64
    assert list(df.columns) == list(dummy_df.columns)


def test_print_summary_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify formatting of the audit summary table."""
    dummy_df = _make_dummy_raw_df()
    out_csv = tmp_path / "summary.csv"
    dummy_df.to_csv(out_csv, index=False)

    print_summary_table(dummy_df, out_csv, "mockhash123")
    captured = capsys.readouterr()

    assert "CQI Arabica Preprocessed Dataset Export Summary" in captured.out
    assert "mockhash123" in captured.out
    assert "Total Valid Records  : 2" in captured.out


def test_main_success(tmp_path: Path) -> None:
    """Verify main entrypoint success return code."""
    raw_csv = tmp_path / "raw.csv"
    out_csv = tmp_path / "out.csv"
    _make_dummy_raw_df().to_csv(raw_csv, index=False)

    exit_code = main(["--raw-path", str(raw_csv), "--output-path", str(out_csv)])
    assert exit_code == 0
    assert out_csv.exists()


def test_main_file_not_found(tmp_path: Path) -> None:
    """Verify handling of missing input files."""
    missing_file = tmp_path / "nonexistent.csv"
    exit_code = main(["--raw-path", str(missing_file)])
    assert exit_code == 1


def test_main_unexpected_error() -> None:
    """Verify graceful handling of unexpected exceptions."""
    with patch("scripts.export_processed_dataset.export_processed_dataset") as mock_export:
        mock_export.side_effect = RuntimeError("Disk write failed")
        exit_code = main([])
        assert exit_code == 2
