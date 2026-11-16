"""Unit tests for data sanitization, outlier filtering, imputation, and splitting."""

import numpy as np
import pandas as pd
import pytest

from coffee_quality.data.cleaner import clean_dataset, split_dataset
from coffee_quality.data.loader import load_raw_data


def test_clean_dataset_real_data() -> None:
    """Verify that cleaning the project's real raw dataset removes duplicates and preserves clean rows."""
    raw_df, _ = load_raw_data()
    clean_df = clean_dataset(raw_df)

    # 1310 raw records minus 47 identical duplicates yields 1263 unique valid records
    assert len(clean_df) == 1263
    assert clean_df.isnull().sum().sum() == 0
    assert (clean_df["altitude_mean_meters"] <= 4000.0).all()
    assert (clean_df["altitude_mean_meters"] >= 0.0).all()
    assert (clean_df["Moisture"] <= 1.0).all()
    assert (clean_df["Moisture"] >= 0.0).all()
    assert set(clean_df["Is_Specialty"].unique()) == {0, 1}


def test_clean_dataset_empty_dataframe() -> None:
    """Verify that attempting to clean an empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Cannot clean an empty DataFrame"):
        clean_dataset(pd.DataFrame())


def test_clean_dataset_no_valid_records_remaining() -> None:
    """Verify that when all rows fail validation, a ValueError is raised."""
    all_invalid = pd.DataFrame(
        {
            "Country.of.Origin": ["Ethiopia"],
            "Grading.Date": ["2015-04-04"],
            "altitude_mean_meters": [9999.0],  # Outlier > 4000m
            "Is_Specialty": [1],
        }
    )
    with pytest.raises(ValueError, match="contains no valid records"):
        clean_dataset(all_invalid)


def test_clean_dataset_synthetic_dirty_data() -> None:
    """Verify imputation, filtering, and sanitization across diverse edge cases."""
    dirty_df = pd.DataFrame(
        {
            "Country.of.Origin": ["Ethiopia", None, "  ", "Guatemala", "Kenya", "Brazil"],
            "Variety": [None, "Bourbon", "Typica", "", "SL28", "Catuai"],
            "Processing.Method": ["Washed / Wet", None, "Natural / Dry", "Unknown", None, "Natural / Dry"],
            "Color": ["Green", None, "Bluish-Green", "Unknown", "Green", None],
            "altitude_mean_meters": [1500.0, None, -50.0, 5000.0, 1600.0, 1200.0],
            "Moisture": [0.12, None, 1.5, -0.1, 0.10, 0.11],
            "Category.One.Defects": [0, None, 1, 0, -2, 0],
            "Category.Two.Defects": [1, None, 0, 0, 0, 2],
            "Quakers": [0, None, 0, 0, 0, 0],
            "Grading.Date": ["2015-04-04", "2015-05-10", "bad-date", "2016-01-01", "2016-02-02", "2016-03-03"],
            "Is_Specialty": [1, 0, 1, 1, 1, None],
        }
    )

    clean_df = clean_dataset(dirty_df)

    # Row 0: valid -> kept
    # Row 1: target=0, missing altitude imputed with median (1500) -> kept
    # Row 2: negative altitude (-50), moisture (1.5), bad date -> dropped
    # Row 3: altitude 5000 -> dropped
    # Row 4: negative defects (-2) -> dropped
    # Row 5: target is None -> dropped
    assert len(clean_df) == 2

    # Check imputations on surviving rows
    assert clean_df.loc[clean_df["Is_Specialty"] == 0, "Country.of.Origin"].iloc[0] == "Unknown"
    assert clean_df.loc[clean_df["Is_Specialty"] == 0, "Variety"].iloc[0] == "Bourbon"
    assert clean_df.loc[clean_df["Is_Specialty"] == 0, "Category.One.Defects"].iloc[0] == 0


def test_clean_dataset_inference_features_without_target() -> None:
    """Verify cleaning unlabelled feature inputs without target column as used during inference."""
    unlabelled_df = pd.DataFrame(
        {
            "Country.of.Origin": ["Ethiopia", None],
            "Variety": [None, "Typica"],
            "Processing.Method": ["Washed / Wet", "Natural / Dry"],
            "Color": ["Green", "Blue-Green"],
            "altitude_mean_meters": [1800.0, 1600.0],
            "Moisture": [0.11, 0.12],
            "Category.One.Defects": [0, 1],
            "Category.Two.Defects": [1, 2],
            "Quakers": [0, 0],
            "Grading.Date": ["2023-01-01", "2023-02-02"],
        }
    )
    cleaned = clean_dataset(unlabelled_df)
    assert len(cleaned) == 2
    assert "Is_Specialty" not in cleaned.columns
    assert cleaned["Country.of.Origin"].iloc[1] == "Unknown"


def test_clean_dataset_partial_columns_subset() -> None:
    """Verify that clean_dataset handles partial column inputs gracefully."""
    partial_df = pd.DataFrame(
        {
            "Country.of.Origin": ["Brazil"],
            "Moisture": [0.11],
        }
    )
    cleaned = clean_dataset(partial_df)
    assert len(cleaned) == 1
    assert "altitude_mean_meters" not in cleaned.columns


def test_clean_dataset_all_continuous_missing() -> None:
    """Verify fallback defaults when continuous columns contain only nulls."""
    df_all_null_continuous = pd.DataFrame(
        {
            "Country.of.Origin": ["Ethiopia", "Guatemala"],
            "altitude_mean_meters": [np.nan, np.nan],
            "Moisture": [np.nan, np.nan],
            "Is_Specialty": [1, 0],
        }
    )
    clean_df = clean_dataset(df_all_null_continuous, default_altitude=1310.6)

    assert (clean_df["altitude_mean_meters"] == 1310.6).all()
    assert (clean_df["Moisture"] == 0.10).all()


def test_clean_dataset_deduplication() -> None:
    """Verify that exact duplicate rows are removed."""
    duplicate_df = pd.DataFrame(
        {
            "Country.of.Origin": ["Ethiopia", "Ethiopia"],
            "Variety": ["Typica", "Typica"],
            "altitude_mean_meters": [1500.0, 1500.0],
            "Moisture": [0.12, 0.12],
            "Is_Specialty": [1, 1],
        }
    )
    clean_df = clean_dataset(duplicate_df)
    assert len(clean_df) == 1


def test_split_dataset_stratification() -> None:
    """Verify that split_dataset preserves proportional target class distribution."""
    raw_df, _ = load_raw_data()
    clean_df = clean_dataset(raw_df)

    train_df, test_df = split_dataset(clean_df, test_size=0.2, random_state=42)

    assert len(train_df) + len(test_df) == len(clean_df)
    assert abs(len(test_df) - int(len(clean_df) * 0.2)) <= 1

    orig_ratio = clean_df["Is_Specialty"].mean()
    train_ratio = train_df["Is_Specialty"].mean()
    test_ratio = test_df["Is_Specialty"].mean()

    assert abs(train_ratio - orig_ratio) < 0.01
    assert abs(test_ratio - orig_ratio) < 0.01


def test_split_dataset_missing_target() -> None:
    """Verify that missing target column raises ValueError."""
    df_no_target = pd.DataFrame({"Country.of.Origin": ["Ethiopia", "Guatemala"]})
    with pytest.raises(ValueError, match="required for stratified split"):
        split_dataset(df_no_target)


def test_split_dataset_insufficient_samples() -> None:
    """Verify that datasets with fewer than 2 records raise ValueError."""
    single_record = pd.DataFrame({"Is_Specialty": [1]})
    with pytest.raises(ValueError, match="at least 2 records"):
        split_dataset(single_record)


def test_split_dataset_single_class() -> None:
    """Verify that target with only one class raises ValueError."""
    single_class_df = pd.DataFrame({"Is_Specialty": [1, 1, 1, 1]})
    with pytest.raises(ValueError, match="at least two distinct target classes"):
        split_dataset(single_class_df)
