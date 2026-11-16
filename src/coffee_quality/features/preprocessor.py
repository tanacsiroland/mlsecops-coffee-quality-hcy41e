"""Feature extraction and preprocessing pipeline module.

Constructs scikit-learn transformers for categorical encoding, continuous and
discrete feature scaling, and date component extraction, free from data leakage.
"""

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

CATEGORICAL_FEATURES: tuple[str, ...] = (
    "Country.of.Origin",
    "Variety",
    "Processing.Method",
    "Color",
)

NUMERICAL_CONTINUOUS_FEATURES: tuple[str, ...] = (
    "altitude_mean_meters",
    "Moisture",
)

NUMERICAL_DISCRETE_FEATURES: tuple[str, ...] = (
    "Category.One.Defects",
    "Category.Two.Defects",
    "Quakers",
)

DATE_FEATURE: str = "Grading.Date"


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transform ISO date strings into numerical year and month feature columns."""

    def fit(self, X: Any, y: Any = None) -> "DateFeatureExtractor":
        """No-op fit method satisfying the TransformerMixin interface."""
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Extract year and month numbers as a 2D float array."""
        if isinstance(X, pd.DataFrame):
            series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        else:
            series = pd.Series(X.ravel() if hasattr(X, "ravel") else X)

        parsed_dates = pd.to_datetime(series, errors="coerce")
        years = parsed_dates.dt.year.fillna(2015).astype(float).to_numpy()
        months = parsed_dates.dt.month.fillna(6).astype(float).to_numpy()

        return np.column_stack([years, months])

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return names of generated date features."""
        return np.array(["grading_year", "grading_month"])


def build_feature_preprocessor() -> ColumnTransformer:
    """Build the Scikit-learn ColumnTransformer encoding categoricals and scaling numerics.

    Returns:
        Unfitted ColumnTransformer instance ready for pipeline integration.
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    cont_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    defect_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", RobustScaler()),
        ]
    )

    date_pipeline = Pipeline(
        steps=[
            ("extractor", DateFeatureExtractor()),
            ("scaler", RobustScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, list(CATEGORICAL_FEATURES)),
            ("cont", cont_pipeline, list(NUMERICAL_CONTINUOUS_FEATURES)),
            ("defect", defect_pipeline, list(NUMERICAL_DISCRETE_FEATURES)),
            ("date", date_pipeline, [DATE_FEATURE]),
        ],
        remainder="drop",
    )

    return preprocessor


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate required predictive feature columns from input DataFrame.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame containing strictly the ordered feature columns.

    Raises:
        ValueError: If any required feature column is missing.
    """
    feature_cols = [
        *CATEGORICAL_FEATURES,
        *NUMERICAL_CONTINUOUS_FEATURES,
        *NUMERICAL_DISCRETE_FEATURES,
        DATE_FEATURE,
    ]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {sorted(missing)}")

    return cast(pd.DataFrame, df[feature_cols].copy())


def extract_features_and_target(
    df: pd.DataFrame,
    target_column: str = "Is_Specialty",
) -> tuple[pd.DataFrame, pd.Series]:
    """Isolate predictive features X and binary target series y.

    Args:
        df: Cleaned dataset containing features and target column.
        target_column: Name of the target column (default 'Is_Specialty').

    Returns:
        Tuple of (X DataFrame, y Series).

    Raises:
        ValueError: If target column is missing from the dataset.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    X = extract_features(df)
    target_series = cast(pd.Series, df[target_column])
    y = target_series.copy().astype(int)

    return X, y
