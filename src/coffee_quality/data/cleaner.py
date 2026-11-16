"""Data cleaning, sanitization, outlier filtering, and dataset splitting module."""

import math
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "Country.of.Origin",
    "Variety",
    "Processing.Method",
    "Color",
)

DISCRETE_DEFECT_COLUMNS: tuple[str, ...] = (
    "Category.One.Defects",
    "Category.Two.Defects",
    "Quakers",
)

CONTINUOUS_COLUMNS: tuple[str, ...] = (
    "altitude_mean_meters",
    "Moisture",
)

TARGET_COLUMN: str = "Is_Specialty"


def clean_dataset(
    df: pd.DataFrame,
    max_altitude: float = 4000.0,
    default_altitude: float = 1310.6,
) -> pd.DataFrame:
    """Sanitize raw tabular coffee lot records per domain rules and curation guidelines.

    Args:
        df: Raw input DataFrame.
        max_altitude: Upper boundary for Arabica elevation (meters). Outliers above are filtered.
        default_altitude: Fallback median elevation if altitude column is entirely null.

    Returns:
        Cleaned, deduplicated, and typed DataFrame.

    Raises:
        ValueError: If input DataFrame is empty or contains no valid rows after cleaning.
    """
    if df.empty:
        raise ValueError("Cannot clean an empty DataFrame.")

    clean_df = df.copy()

    # Drop records missing target label
    if TARGET_COLUMN in clean_df.columns:
        clean_df = clean_df.dropna(subset=[TARGET_COLUMN])
        valid_targets = clean_df[TARGET_COLUMN].isin([0, 1])
        clean_df = cast(pd.DataFrame, clean_df.loc[valid_targets])
        clean_df[TARGET_COLUMN] = clean_df[TARGET_COLUMN].astype(int)

    # Impute categorical attributes with 'Unknown'
    for col in CATEGORICAL_COLUMNS:
        if col in clean_df.columns:
            series = clean_df[col].fillna("Unknown").astype(str).str.strip()
            clean_df[col] = series.replace("", "Unknown")

    # Impute discrete defect counts with 0
    for col in DISCRETE_DEFECT_COLUMNS:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna(0).round().astype(int)
            clean_df = cast(pd.DataFrame, clean_df.loc[clean_df[col] >= 0])

    # Impute continuous attributes with median values
    for col in CONTINUOUS_COLUMNS:
        if col in clean_df.columns:
            col_series = cast(pd.Series, clean_df[col])
            raw_median = col_series.median()
            median_val = float(cast(float, raw_median))
            if math.isnan(median_val):
                median_val = default_altitude if col == "altitude_mean_meters" else 0.10
            clean_df[col] = col_series.fillna(median_val).astype(float)

    # Filter altitude physical outliers
    if "altitude_mean_meters" in clean_df.columns:
        valid_altitude = (clean_df["altitude_mean_meters"] >= 0.0) & (
            clean_df["altitude_mean_meters"] <= max_altitude
        )
        clean_df = cast(pd.DataFrame, clean_df.loc[valid_altitude])

    # Filter moisture outliers
    if "Moisture" in clean_df.columns:
        valid_moisture = (clean_df["Moisture"] >= 0.0) & (clean_df["Moisture"] <= 1.0)
        clean_df = cast(pd.DataFrame, clean_df.loc[valid_moisture])

    # Normalize grading date to ISO 8601 string (YYYY-MM-DD)
    if "Grading.Date" in clean_df.columns:
        parsed_dates = pd.to_datetime(clean_df["Grading.Date"], errors="coerce")
        valid_date_mask = parsed_dates.notna()
        clean_df = cast(pd.DataFrame, clean_df.loc[valid_date_mask])
        formatted_dates = parsed_dates.loc[valid_date_mask].dt.strftime("%Y-%m-%d")
        clean_df["Grading.Date"] = formatted_dates

    # Deduplicate records
    clean_df = clean_df.drop_duplicates().reset_index(drop=True)

    if clean_df.empty:
        raise ValueError("DataFrame contains no valid records after sanitization.")

    return clean_df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into stratified training and testing sets.

    Args:
        df: Cleaned dataset containing features and target column.
        test_size: Proportion of dataset allocated to evaluation split (default 0.2).
        random_state: Seed for deterministic sampling.

    Returns:
        Tuple of (train_df, test_df) with preserved target class distribution.

    Raises:
        ValueError: If target column is missing or target has fewer than 2 classes.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required for stratified split.")

    if len(df) < 2:
        raise ValueError("Dataset must contain at least 2 records to split.")

    target_series = cast(pd.Series, df[TARGET_COLUMN])
    unique_classes = len(target_series.unique())
    if unique_classes < 2:
        raise ValueError("Stratified split requires at least two distinct target classes.")

    train_raw, test_raw = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )

    train_df = cast(pd.DataFrame, train_raw).reset_index(drop=True)
    test_df = cast(pd.DataFrame, test_raw).reset_index(drop=True)

    return train_df, test_df
