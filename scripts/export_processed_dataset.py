"""Export preprocessed, leak-free Arabica coffee dataset for Moodle submission package.

This script executes the complete data ingestion and cleaning pipeline on raw coffee lot records,
filters physical outliers, binarizes the target specialty label, and exports a clean CSV.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import cast

import pandas as pd

from coffee_quality.config import get_settings
from coffee_quality.data.cleaner import clean_dataset
from coffee_quality.data.loader import compute_file_sha256, load_raw_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dataset-exporter")

DEFAULT_OUTPUT_PATH = Path("data/processed/arabica_cleaned_specialty.csv")


def export_processed_dataset(
    raw_path: Path | None = None,
    output_path: Path | None = None,
    max_altitude: float = 4000.0,
) -> tuple[pd.DataFrame, str]:
    """Load raw dataset, sanitize and filter outliers, and write clean CSV.

    Args:
        raw_path: Path to the raw input CSV. Defaults to configured data_raw_path.
        output_path: Destination path for the processed CSV. Defaults to DEFAULT_OUTPUT_PATH.
        max_altitude: Upper boundary for Arabica elevation in meters.

    Returns:
        Tuple containing the processed DataFrame and its cryptographic SHA-256 digest.

    Raises:
        FileNotFoundError: If the raw dataset does not exist.
        ValueError: If cleaned dataset contains no valid records.
    """
    settings = get_settings()
    source_path = raw_path or settings.data_raw_path
    dest_path = output_path or DEFAULT_OUTPUT_PATH

    logger.info("Ingesting raw coffee quality dataset from: %s", source_path.resolve())
    raw_df, raw_sha256 = load_raw_data(source_path)
    logger.info("Raw dataset loaded: %d rows, %d columns (SHA-256: %s)", len(raw_df), len(raw_df.columns), raw_sha256)

    logger.info("Sanitizing records, filtering elevation > %.1f m, and imputing values...", max_altitude)
    clean_df = clean_dataset(raw_df, max_altitude=max_altitude)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(dest_path, index=False, encoding="utf-8")

    out_sha256 = compute_file_sha256(dest_path)
    logger.info("Clean dataset exported successfully to: %s", dest_path.resolve())
    logger.info("Output SHA-256 checksum: %s", out_sha256)

    return clean_df, out_sha256


def print_summary_table(df: pd.DataFrame, output_path: Path, sha256_hash: str) -> None:
    """Print structured audit summary of the exported dataset."""
    specialty_series = cast(pd.Series, df["Is_Specialty"])
    total_records = len(df)
    specialty_count = int(specialty_series.sum())
    commercial_count = total_records - specialty_count
    specialty_ratio = (specialty_count / total_records) * 100.0 if total_records > 0 else 0.0

    print("\n" + "=" * 70)
    print("  CQI Arabica Preprocessed Dataset Export Summary")
    print("=" * 70)
    print(f"  Target File          : {output_path.resolve()}")
    print(f"  File Size            : {output_path.stat().st_size:,} bytes")
    print(f"  Cryptographic Hash   : SHA-256 {sha256_hash}")
    print(f"  Total Valid Records  : {total_records:,}")
    print(f"  Feature Columns (10) : {', '.join([c for c in df.columns if c != 'Is_Specialty'])}")
    print("  Target Column        : Is_Specialty (1 = Specialty, 0 = Commercial)")
    print(f"  Class Distribution   : Specialty = {specialty_count:,} ({specialty_ratio:.1f}%) | "
          f"Commercial = {commercial_count:,} ({100.0 - specialty_ratio:.1f}%)")
    print("=" * 70 + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for dataset exporter."""
    parser = argparse.ArgumentParser(
        description="Export preprocessed, sanitized Arabica coffee dataset for Moodle package.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Path to raw CSV dataset (default: configured data_raw_path)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--max-altitude",
        type=float,
        default=4000.0,
        help="Upper altitude threshold in meters (default: 4000.0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution entrypoint."""
    args = parse_args(argv)
    try:
        clean_df, file_hash = export_processed_dataset(
            raw_path=args.raw_path,
            output_path=args.output_path,
            max_altitude=args.max_altitude,
        )
        print_summary_table(clean_df, args.output_path, file_hash)
        return 0
    except FileNotFoundError as exc:
        logger.error("Dataset file error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error during dataset export: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
