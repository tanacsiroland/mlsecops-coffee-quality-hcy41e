"""Unit tests for feature preprocessing, extraction, and transformation pipeline."""

import numpy as np
import pandas as pd
import pytest

from coffee_quality.data.cleaner import clean_dataset, split_dataset
from coffee_quality.data.loader import load_raw_data
from coffee_quality.features.preprocessor import (
    DateFeatureExtractor,
    build_feature_preprocessor,
    extract_features,
    extract_features_and_target,
)


def test_date_feature_extractor_dataframe() -> None:
    """Verify DateFeatureExtractor converts DataFrame date strings to year and month columns."""
    df = pd.DataFrame({"Grading.Date": ["2015-04-04", "2018-11-20", "invalid-date"]})
    extractor = DateFeatureExtractor()

    fitted = extractor.fit(df)
    assert fitted is extractor

    result = extractor.transform(df)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    assert result[0, 0] == 2015.0
    assert result[0, 1] == 4.0
    assert result[1, 0] == 2018.0
    assert result[1, 1] == 11.0
    # Invalid date falls back to default 2015, 6
    assert result[2, 0] == 2015.0
    assert result[2, 1] == 6.0

    feature_names = extractor.get_feature_names_out()
    assert list(feature_names) == ["grading_year", "grading_month"]


def test_date_feature_extractor_series_and_list() -> None:
    """Verify DateFeatureExtractor handles Series and array inputs."""
    series = pd.Series(["2017-06-15"])
    extractor = DateFeatureExtractor()
    res_series = extractor.transform(series)
    assert res_series[0, 0] == 2017.0
    assert res_series[0, 1] == 6.0

    raw_list = ["2019-12-01"]
    res_list = extractor.transform(raw_list)
    assert res_list[0, 0] == 2019.0
    assert res_list[0, 1] == 12.0


def test_preprocessor_pipeline_real_data() -> None:
    """Verify fitting ColumnTransformer on real train split and transforming test split."""
    raw_df, _ = load_raw_data()
    clean_df = clean_dataset(raw_df)
    train_df, test_df = split_dataset(clean_df, test_size=0.2, random_state=42)

    X_train, y_train = extract_features_and_target(train_df)
    X_test, y_test = extract_features_and_target(test_df)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    preprocessor = build_feature_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    assert isinstance(X_train_transformed, np.ndarray)
    assert isinstance(X_test_transformed, np.ndarray)
    assert X_train_transformed.ndim == 2
    assert X_test_transformed.ndim == 2
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    assert np.isnan(X_train_transformed).sum() == 0
    assert np.isnan(X_test_transformed).sum() == 0


def test_preprocessor_unseen_categories() -> None:
    """Verify that unseen categories during inference are safely ignored (encoded as 0s)."""
    train_df = pd.DataFrame(
        {
            "Country.of.Origin": ["Ethiopia", "Guatemala"],
            "Variety": ["Typica", "Bourbon"],
            "Processing.Method": ["Washed / Wet", "Natural / Dry"],
            "Color": ["Green", "Blue-Green"],
            "altitude_mean_meters": [1500.0, 1600.0],
            "Moisture": [0.11, 0.12],
            "Category.One.Defects": [0, 1],
            "Category.Two.Defects": [2, 0],
            "Quakers": [0, 0],
            "Grading.Date": ["2015-04-04", "2016-05-05"],
            "Is_Specialty": [1, 0],
        }
    )

    X_train, _ = extract_features_and_target(train_df)
    preprocessor = build_feature_preprocessor()
    preprocessor.fit(X_train)

    unseen_inference_df = pd.DataFrame(
        {
            "Country.of.Origin": ["Atlantis"],
            "Variety": ["AlienVariety"],
            "Processing.Method": ["SuperProcessing"],
            "Color": ["NeonYellow"],
            "altitude_mean_meters": [2000.0],
            "Moisture": [0.10],
            "Category.One.Defects": [0],
            "Category.Two.Defects": [1],
            "Quakers": [0],
            "Grading.Date": ["2025-01-01"],
        }
    )

    X_unseen = extract_features(unseen_inference_df)
    transformed = preprocessor.transform(X_unseen)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 1
    assert np.isnan(transformed).sum() == 0


def test_extract_features_missing_column() -> None:
    """Verify extract_features raises ValueError when required column is absent."""
    incomplete = pd.DataFrame({"Country.of.Origin": ["Ethiopia"]})
    with pytest.raises(ValueError, match="missing required feature columns"):
        extract_features(incomplete)


def test_extract_features_and_target_missing_target() -> None:
    """Verify extract_features_and_target raises ValueError when target is absent."""
    valid_features = pd.DataFrame(
        {
            "Country.of.Origin": ["Ethiopia"],
            "Variety": ["Typica"],
            "Processing.Method": ["Washed / Wet"],
            "Color": ["Green"],
            "altitude_mean_meters": [1500.0],
            "Moisture": [0.11],
            "Category.One.Defects": [0],
            "Category.Two.Defects": [1],
            "Quakers": [0],
            "Grading.Date": ["2015-04-04"],
        }
    )
    with pytest.raises(ValueError, match="missing from the dataset"):
        extract_features_and_target(valid_features)
